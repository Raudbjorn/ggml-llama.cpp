#!/bin/sh
# Check for backwards-incompatible API and ABI changes between two builds
#
# Backwards-incompatible API changes, such as removing a value from an enum,
# are checked by abi-compliance-checker. Such changes can break compilation of
# existing programs.
#
# Backwards-incompatible ABI changes, such as the removal of a public function,
# are checked by libabigail-tools. Such changes could break run-time dynamic
# linking and type layouts. Complete type checks require unstripped DWARF.
#
# Commands:
#   --generate <build-dir>: Creates API/ABI dumps in <build-dir>
#                           <build-dir> is expected to be a CMake build result
#   --check <dir1> <dir2>:  Compares dumps in <dir1> and <dir2>
#                           Comparison exit codes
#                              0: all good
#                              1: incompatible changes or incomplete/failed checks
#
# Options:
#   --include-path <dir>:   a-c-c calls gcc on headers; use this option to add
#                           directories to gcc's search path
#
#
# This script would typically be used before cutting a release:
#
#   1. Generate API/ABI dump for the old version
#
#      $ check-apiabi-compat.sh --generate <build-dir-old> libfoo [ libbar ...]
#
#   2. <update source>
#
#   3. Generate API/ABI dump for the new version
#
#      $ check-apiabi-compat.sh --generate <build-dir-new> libfoo [ libbar ...]
#
#   4. Compare the two dumps
#
#      $ check-apiabi-compat.sh --check <old-build-dir> <new-build-dir>
#
#      Exit 0 covers the supplied dumps/manifest only. Nonzero means a
#      reported incompatibility, a tool failure, or incomplete evidence.
#      C++ deltas require explicit review: bit 4 can include layout breaks.
set -eu

# Preconditions
if ! command -v abi-compliance-checker >/dev/null 2>&1; then
    echo "abi-compliance-checker is not installed." >&2
    exit 1
elif ! command -v abidw >/dev/null 2>&1 || ! command -v abidiff >/dev/null 2>&1; then
    echo "abigail-tools are not installed." >&2
    exit 1
fi

# Some generic functions
usage() {
    echo "Usage: $0 [ --include-path <dir> ] --generate <build-dir> libXXX [ libYYY ... ]" >&2
    echo "       $0 [ --strict ] --check <old-build-dir> <new-build-dir>" >&2
    echo "" >&2
    echo "  --cxx --type-manifest <json>: unsuppressed C++ ABI and explicit type-coverage gate" >&2
    echo "  --headers-dir <dir>: header tree for source API checks (default: include)" >&2
    echo "  --strict: fail on any API/ABI change, including backwards-compatible additions" >&2
}

get_cmake_project_name() {
    sed -nr 's/^project\("(.*)".*$/\1/p' CMakeLists.txt
}

get_cmake_version() {
    major="$(sed -nr 's/^set\([A-Z]+_VERSION_MAJOR ([0-9]+)\)$/\1/p' CMakeLists.txt)"
    minor="$(sed -nr 's/^set\([A-Z]+_VERSION_MINOR ([0-9]+)\)$/\1/p' CMakeLists.txt)"
    patch="$(sed -nr 's/^set\([A-Z]+_VERSION_PATCH ([0-9]+)\)$/\1/p' CMakeLists.txt)"
    echo "$major.$minor.$patch"
}

# Option parsing and validation
DO_GEN=0
DO_CHECK=0
STRICT=0
CXX=0
TYPE_MANIFEST=
HEADERS_DIR=include
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD_DIR=
BUILD_DIR_NEW=
INCLUDE_PATHS=
while [ "$#" -gt 0 ]; do
    case "$1" in
    --generate=*)
        DO_GEN=1
        BUILD_DIR="${1#*=}"
        shift
        ;;
    --generate)
        DO_GEN=1
        if [ -z "${2:-}" ]; then
            usage
            exit 1
        fi
        BUILD_DIR="$2"
        shift 2
        ;;
    --check)
        DO_CHECK=1
        if [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
            usage
            exit 1
        elif ! [ -d "$2" ]; then
            echo "$2 is not a directory." >&2
            exit 1
        elif ! [ -d "$3" ]; then
            echo "$3 is not a directory." >&2
            exit 1
        fi
        BUILD_DIR="$2"
        BUILD_DIR_NEW="$3"
        shift 3
        ;;
    --include-path=*)
        INCLUDE_PATHS="$INCLUDE_PATHS ${1#*=}"
        shift
        ;;
    --include-path)
        if [ -z "${2:-}" ]; then
            usage
            exit 1
        fi
        INCLUDE_PATHS="$INCLUDE_PATHS $2"
        shift 2
        ;;
    --cxx)
        CXX=1
        shift
        ;;
    --type-manifest | --headers-dir)
        if [ -z "${2:-}" ]; then usage; exit 1; fi
        case "$1" in
            --type-manifest) TYPE_MANIFEST="$2" ;;
            --headers-dir) HEADERS_DIR="$2" ;;
        esac
        shift 2
        ;;
    --strict)
        STRICT=1
        shift
        ;;
    -h | --help)
        usage
        exit 1
        ;;
    -?*)
        usage
        exit 1
        ;;
    *)
        break
        ;;
    esac
done
if [ $((DO_GEN + DO_CHECK)) -ne 1 ]; then
    echo "Use exactly one --command." >&2
    exit 1
fi
if [ "$CXX" -eq 0 ] && [ -n "$TYPE_MANIFEST" ]; then
    echo "--type-manifest requires --cxx." >&2
    exit 1
fi
if [ "$CXX" -eq 1 ] && [ "$DO_GEN" -eq 1 ] && ! [ -f "$TYPE_MANIFEST" ]; then
    echo "--cxx generation requires a reviewed --type-manifest file." >&2
    exit 1
fi
PROJECT_NAME="$(get_cmake_project_name)"
PROJECT_VERSION="$(get_cmake_version)"
LIB_NAMES=""
while [ "$#" -gt 0 ]; do
    if [ "${1#lib}" = "$1" ]; then
        echo "Library to check must start with libXXX." >&2
        exit 1
    fi
    LIB_NAMES="$LIB_NAMES $1"
    shift
done

dump_current_api() {
    echo "Dumping API..."

    DESCRIPTOR="$BUILD_DIR/apiabi/acc-descriptor.xml"
    mkdir -p "$BUILD_DIR/apiabi"
    cat >"$DESCRIPTOR" <<EOF
<version>$PROJECT_VERSION</version>
<headers>$HEADERS_DIR</headers>
<add_include_paths>$INCLUDE_PATHS</add_include_paths>
EOF

    # This addresses a bug between a-c-c and universal-ctags, manifested when
    # a name is use both for a tag and a function name
    mkdir -p "$BUILD_DIR/apiabi/ctags"
    echo "--fields=-t" >"$BUILD_DIR/apiabi/ctags/acc.ctags"

    XDG_CONFIG_HOME="$BUILD_DIR/apiabi" abi-compliance-checker \
        -headers-only \
        -lib "$PROJECT_NAME" \
        -dump "$DESCRIPTOR" \
        -log-path "$BUILD_DIR/apiabi/acc.log" \
        -dump-path "$BUILD_DIR/apiabi/api.dump"
    # acc generates this file with an ancient timestamp, which confuses gzip
    touch "$BUILD_DIR/apiabi/api.dump"
}

dump_current_abi() {
    echo "Dumping ABIs ..."
    mkdir -p "$BUILD_DIR/apiabi"
    # The suppressions are needed to avoid including all the internal C++
    # symbols, and system types
    if [ "$CXX" -eq 0 ]; then
    cat >"$BUILD_DIR/apiabi/abidw.suppress" <<EOF
[suppress_function]

label = suppress internal C++ mangled functions
symbol_name_regexp = ^_Z
drop = yes

[suppress_variable]
label = suppress internal C++ mangled variables
symbol_name_regexp = ^_Z
drop = yes

[suppress_type]
label = Suppress types outside of our own source
source_location_not_regexp = ^include/
drop = yes
EOF
    else
        cp "$TYPE_MANIFEST" "$BUILD_DIR/apiabi/cxx-type-manifest.json"
    fi

    # In abidw 2.5, handling of undefined stuff was changed a bit
    abidw_version="$(abidw --version | sed -nr 's/^abidw:?[[:space:]]+([0-9]+\.[0-9]+).*$/\1/p')"
    if [ -z "$abidw_version" ]; then
        echo "Cannot parse abidw version." >&2
        exit 1
    fi
    abidw_major="${abidw_version%.*}"
    abidw_minor="${abidw_version#*.}"
    if [ "$abidw_major" -gt 2 ] || { [ "$abidw_major" -eq 2 ] && [ "$abidw_minor" -gt 4 ]; }; then
        abidw_undefined_syms_options="--no-load-undefined-interfaces"
    else
        abidw_undefined_syms_options="--drop-undefined-syms"
    fi

    for lib_name in $LIB_NAMES; do
        # Depending on where add_library resides, the libraries can end up in
        # build/src or build/bin
        lib_path="$BUILD_DIR/src/$lib_name.so"
        if ! [ -f "$lib_path" ]; then
            lib_path="$BUILD_DIR/bin/$lib_name.so"
            if ! [ -f "$lib_path" ]; then
                echo "Cannot find library $lib_name.so" >&2
                exit 1
            fi
        fi
        if [ "$CXX" -eq 1 ]; then
            abidw --fail-no-debug-info --load-all-types --show-locs \
                "$abidw_undefined_syms_options" \
                --out-file "$BUILD_DIR/apiabi/$lib_name.abi.xml" "$lib_path"
            python3 "$SCRIPT_DIR/check-apiabi-types.py" \
                "$BUILD_DIR/apiabi/cxx-type-manifest.json" "$lib_name" \
                "$BUILD_DIR/apiabi/$lib_name.abi.xml"
        else
            abidw --headers-dir "$HEADERS_DIR" \
                "$abidw_undefined_syms_options" \
                --suppressions "$BUILD_DIR/apiabi/abidw.suppress" \
                --out-file "$BUILD_DIR/apiabi/$lib_name.abi.xml" "$lib_path"
        fi
    done
}

# Run the actual commands
if [ "$DO_GEN" -eq 1 ]; then
    if [ -z "$LIB_NAMES" ]; then echo "No libraries requested." >&2; exit 1; fi
    rm -f "$BUILD_DIR/apiabi/mode"
    if [ "$CXX" -eq 1 ]; then
        python3 "$SCRIPT_DIR/check-apiabi-types.py" "$TYPE_MANIFEST" --requested-libraries "$LIB_NAMES"
    fi
    dump_current_api
    dump_current_abi
    if [ "$CXX" -eq 1 ]; then
        python3 "$SCRIPT_DIR/check-apiabi-types.py" "$BUILD_DIR/apiabi/cxx-type-manifest.json" --libraries "$BUILD_DIR/apiabi"
    fi
    if [ "$CXX" -eq 1 ]; then mode=cxx; else mode=c; fi
    printf '%s\n' "$mode" >"$BUILD_DIR/apiabi/mode"
    exit 0
elif [ "$DO_CHECK" -eq 1 ]; then
    if [ "$CXX" -eq 1 ]; then
        for dir in "$BUILD_DIR" "$BUILD_DIR_NEW"; do
            if [ "$(cat "$dir/apiabi/mode" 2>/dev/null)" != cxx ]; then
                echo "Missing successful C++ generation marker: $dir" >&2; exit 1
            fi
            python3 "$SCRIPT_DIR/check-apiabi-types.py" "$dir/apiabi/cxx-type-manifest.json" --libraries "$dir/apiabi"
        done
        cmp "$BUILD_DIR/apiabi/cxx-type-manifest.json" "$BUILD_DIR_NEW/apiabi/cxx-type-manifest.json"
    elif [ "$(cat "$BUILD_DIR/apiabi/mode" 2>/dev/null || true)" = cxx ] ||
         [ "$(cat "$BUILD_DIR_NEW/apiabi/mode" 2>/dev/null || true)" = cxx ]; then
        echo "C++ dumps must be compared with --cxx." >&2; exit 1
    fi
    set -- "$BUILD_DIR/apiabi/"lib*.abi.xml
    if ! [ -f "$1" ]; then echo "No baseline ABI dumps." >&2; exit 1; fi
    # From here on, collect tool failures without mistaking them for compatibility.
    set +e

    abi-compliance-checker \
        -strict \
        -source \
        -library "$PROJECT_NAME" \
        -old "$BUILD_DIR/apiabi/api.dump" \
        -new "$BUILD_DIR_NEW/apiabi/api.dump" \
        -src-report-path "$BUILD_DIR_NEW/apiabi/api_compat_report.html"
    API_RESULT=$?

    ABI_RESULT=0
    for xml_file in "$BUILD_DIR/apiabi/"lib*.abi.xml; do
        xml_file_new="$BUILD_DIR_NEW/apiabi/$(basename "$xml_file")"

        if ! [ -f "$xml_file_new" ]; then
            echo "Cannot compare, missing file: $xml_file_new" >&2
            exit 1
        fi

        if [ "$CXX" -eq 1 ]; then
            lib_name=$(basename "$xml_file" .abi.xml)
            python3 "$SCRIPT_DIR/check-apiabi-types.py" "$BUILD_DIR/apiabi/cxx-type-manifest.json" "$lib_name" "$xml_file" || exit 1
            python3 "$SCRIPT_DIR/check-apiabi-types.py" "$BUILD_DIR_NEW/apiabi/cxx-type-manifest.json" "$lib_name" "$xml_file_new" || exit 1
            abidiff --no-default-suppression --non-reachable-types "$xml_file" "$xml_file_new"
        else
            abidiff "$xml_file" "$xml_file_new"
        fi
        res=$?
        if [ "$((res & 3))" -ne 0 ] || [ "$res" -gt 15 ]; then
            echo "ERROR: abidiff failed (exit $res): $xml_file" >&2
            ABI_RESULT=1
        fi
        [ "$((res & 8))" -ne 0 ] && ABI_RESULT=1
        # Libabigail can report layout breaks with bit 4 alone. C++ mode
        # requires review of every delta, including genuinely additive ones.
        if [ "$CXX" -eq 1 ] || [ "$STRICT" -eq 1 ]; then
            [ "$((res & 4))" -ne 0 ] && ABI_RESULT=1
        fi
    done

    if [ "$API_RESULT" -gt 0 ]; then
        if [ "$STRICT" -eq 1 ]; then
            echo "ERROR: API changed - a minor version bump is required." >&2
        else
            echo "ERROR: API changed with backwards-incompatible problems - a major version bump is required." >&2
        fi
    fi
    if [ "$ABI_RESULT" -gt 0 ]; then
        echo "ERROR: ABI comparison failed or reported changes requiring review; inspect tool status and output." >&2
    fi
    if [ "$((API_RESULT + ABI_RESULT))" -gt 0 ]; then
        exit 1
    fi
fi
