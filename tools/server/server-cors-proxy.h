#pragma once

#include "common.h"
#include "http.h"
#include "server-common.h"
#include "server-http.h"
#include "server-models.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

struct proxy_target_policy_result {
    bool allowed;
    std::string message;
};

static std::string proxy_normalize_host(std::string host) {
    std::transform(host.begin(), host.end(), host.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (!host.empty() && host.back() == '.') {
        host.pop_back();
    }
    return host;
}

static bool proxy_host_is_or_has_suffix(const std::string & host, const std::string & suffix) {
    return host == suffix ||
           (host.size() > suffix.size() &&
            host.compare(host.size() - suffix.size(), suffix.size(), suffix) == 0 &&
            host[host.size() - suffix.size() - 1] == '.');
}

static bool proxy_ipv4_has_prefix(uint32_t address, uint32_t network, uint32_t mask) {
    return (address & mask) == network;
}

static bool proxy_ipv4_is_global(uint32_t address) {
    // IANA protocol-assignment anycasts are globally reachable exceptions
    // inside the otherwise non-global 192.0.0.0/24 range.
    if (address == 0xc0000009U || address == 0xc000000aU) {
        return true;
    }

    return !proxy_ipv4_has_prefix(address, 0x00000000U, 0xff000000U) &&
           !proxy_ipv4_has_prefix(address, 0x0a000000U, 0xff000000U) &&
           !proxy_ipv4_has_prefix(address, 0x64400000U, 0xffc00000U) &&
           !proxy_ipv4_has_prefix(address, 0x7f000000U, 0xff000000U) &&
           !proxy_ipv4_has_prefix(address, 0xa9fe0000U, 0xffff0000U) &&
           !proxy_ipv4_has_prefix(address, 0xac100000U, 0xfff00000U) &&
           !proxy_ipv4_has_prefix(address, 0xc0000000U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xc0000200U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xc0586300U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xc0a80000U, 0xffff0000U) &&
           !proxy_ipv4_has_prefix(address, 0xc6120000U, 0xfffe0000U) &&
           !proxy_ipv4_has_prefix(address, 0xc6336400U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xcb007100U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xe0000000U, 0xf0000000U) &&
           !proxy_ipv4_has_prefix(address, 0xf0000000U, 0xf0000000U);
}

static bool proxy_ipv6_has_prefix(
        const uint8_t * address,
        const std::array<uint8_t, 16> & prefix,
        size_t prefix_bits) {
    const size_t whole_bytes = prefix_bits / 8;
    const size_t remaining_bits = prefix_bits % 8;
    for (size_t i = 0; i < whole_bytes; ++i) {
        if (address[i] != prefix[i]) {
            return false;
        }
    }
    if (remaining_bits != 0) {
        const uint8_t mask = static_cast<uint8_t>(0xffU << (8 - remaining_bits));
        return (address[whole_bytes] & mask) == (prefix[whole_bytes] & mask);
    }
    return true;
}

static uint32_t proxy_ipv4_from_ipv6_tail(const uint8_t * bytes) {
    return (static_cast<uint32_t>(bytes[12]) << 24) |
           (static_cast<uint32_t>(bytes[13]) << 16) |
           (static_cast<uint32_t>(bytes[14]) << 8) |
           static_cast<uint32_t>(bytes[15]);
}

static bool proxy_ipv6_is_global(const in6_addr & address) {
    static const std::array<uint8_t, 16> IPV4_MAPPED       = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    static const std::array<uint8_t, 16> NAT64_GLOBAL      = {0x00, 0x64, 0xff, 0x9b};
    static const std::array<uint8_t, 16> IANA_SPECIAL      = {0x20, 0x01};
    static const std::array<uint8_t, 16> PCP_ANYCAST       = {0x20, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    static const std::array<uint8_t, 16> TURN_ANYCAST      = {0x20, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};
    static const std::array<uint8_t, 16> DNS_SD_ANYCAST    = {0x20, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3};
    static const std::array<uint8_t, 16> AMT               = {0x20, 0x01, 0x00, 0x03};
    static const std::array<uint8_t, 16> AS112_V6          = {0x20, 0x01, 0x00, 0x04, 0x01, 0x12};
    static const std::array<uint8_t, 16> ORCHID_V2         = {0x20, 0x01, 0x00, 0x20};
    static const std::array<uint8_t, 16> DRONE_REMOTE_ID   = {0x20, 0x01, 0x00, 0x30};
    static const std::array<uint8_t, 16> DOCUMENTATION_V6  = {0x20, 0x01, 0x0d, 0xb8};
    static const std::array<uint8_t, 16> SIX_TO_FOUR       = {0x20, 0x02};
    static const std::array<uint8_t, 16> DOCUMENTATION_V2  = {0x3f, 0xff, 0x00};
    static const std::array<uint8_t, 16> SEGMENT_ROUTING   = {0x5f, 0x00};

    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(&address);

    if (proxy_ipv6_has_prefix(bytes, IPV4_MAPPED, 96)) {
        return proxy_ipv4_is_global(proxy_ipv4_from_ipv6_tail(bytes));
    }
    if (proxy_ipv6_has_prefix(bytes, NAT64_GLOBAL, 96)) {
        return proxy_ipv4_is_global(proxy_ipv4_from_ipv6_tail(bytes));
    }

    // Globally routable unicast space is 2000::/3. Other address families
    // include unspecified, loopback, ULA, link-local, site-local, and multicast.
    if ((bytes[0] & 0xe0U) != 0x20U) {
        return false;
    }

    // The IANA 2001::/23 registry is non-global by default. Only allocations
    // explicitly marked globally reachable are allowed.
    if (proxy_ipv6_has_prefix(bytes, IANA_SPECIAL, 23)) {
        return proxy_ipv6_has_prefix(bytes, PCP_ANYCAST, 128) ||
               proxy_ipv6_has_prefix(bytes, TURN_ANYCAST, 128) ||
               proxy_ipv6_has_prefix(bytes, DNS_SD_ANYCAST, 128) ||
               proxy_ipv6_has_prefix(bytes, AMT, 32) ||
               proxy_ipv6_has_prefix(bytes, AS112_V6, 48) ||
               proxy_ipv6_has_prefix(bytes, ORCHID_V2, 28) ||
               proxy_ipv6_has_prefix(bytes, DRONE_REMOTE_ID, 28);
    }

    return !proxy_ipv6_has_prefix(bytes, DOCUMENTATION_V6, 32) &&
           !proxy_ipv6_has_prefix(bytes, SIX_TO_FOUR, 16) &&
           !proxy_ipv6_has_prefix(bytes, DOCUMENTATION_V2, 20) &&
           !proxy_ipv6_has_prefix(bytes, SEGMENT_ROUTING, 16);
}

static bool proxy_sockaddr_is_global(const sockaddr * address) {
    if (address->sa_family == AF_INET) {
        const auto * ipv4 = reinterpret_cast<const sockaddr_in *>(address);
        return proxy_ipv4_is_global(ntohl(ipv4->sin_addr.s_addr));
    }
    if (address->sa_family == AF_INET6) {
        const auto * ipv6 = reinterpret_cast<const sockaddr_in6 *>(address);
        return proxy_ipv6_is_global(ipv6->sin6_addr);
    }
    return false;
}

static proxy_target_policy_result proxy_target_policy(
        const common_http_url & parsed_url,
        const std::vector<std::string> & allowlist) {
    if (parsed_url.scheme != "http" && parsed_url.scheme != "https") {
        return {false, "unsupported target URL scheme"};
    }
    if (parsed_url.host.empty()) {
        return {false, "target URL is missing a host"};
    }
    if (!parsed_url.user.empty() || !parsed_url.password.empty()) {
        return {false, "userinfo in target URLs is not allowed"};
    }

    const std::string host = proxy_normalize_host(parsed_url.host);
    if (host.empty()) {
        return {false, "target URL is missing a host"};
    }

    for (const std::string & allowed_host : allowlist) {
        if (host == proxy_normalize_host(allowed_host)) {
            return {true, ""};
        }
    }

    if (proxy_host_is_or_has_suffix(host, "localhost") ||
        proxy_host_is_or_has_suffix(host, "local") ||
        proxy_host_is_or_has_suffix(host, "localdomain") ||
        proxy_host_is_or_has_suffix(host, "internal") ||
        proxy_host_is_or_has_suffix(host, "home.arpa")) {
        return {false, "local target host is not allowed"};
    }
    if (host == "metadata.google.internal" || host == "metadata.goog") {
        return {false, "metadata target host is not allowed"};
    }

    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_NUMERICHOST;
    addrinfo * addresses = nullptr;
    const int lookup_result = getaddrinfo(host.c_str(), nullptr, &hints, &addresses);
    if (lookup_result != 0) {
        return {true, ""}; // Hostname: Tier 4a intentionally does not resolve DNS.
    }

    bool global = addresses != nullptr;
    for (const addrinfo * current = addresses; current != nullptr; current = current->ai_next) {
        if (current->ai_addr == nullptr || !proxy_sockaddr_is_global(current->ai_addr)) {
            global = false;
            break;
        }
    }
    freeaddrinfo(addresses);

    return global
            ? proxy_target_policy_result{true, ""}
            : proxy_target_policy_result{false, "non-global numeric target host is not allowed"};
}

static std::string proxy_header_to_lower(std::string header) {
    std::transform(header.begin(), header.end(), header.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return header;
}

static server_http_res_ptr proxy_policy_error(const std::string & message) {
    auto res = std::make_unique<server_http_res>();
    res->status = 400;
    res->data = safe_json_to_str({
        {"error", format_error_response(message, ERROR_TYPE_INVALID_REQUEST)},
    });
    return res;
}

static server_http_res_ptr proxy_request(
        const server_http_req & req,
        const std::string & method,
        const std::vector<std::string> & allowlist) {
    common_http_url parsed_url;
    try {
        parsed_url = common_http_parse_url(req.get_param("url"));
    } catch (const std::exception & e) {
        return proxy_policy_error(std::string("invalid target URL: ") + e.what());
    }

    const proxy_target_policy_result policy = proxy_target_policy(parsed_url, allowlist);
    if (!policy.allowed) {
        return proxy_policy_error(policy.message);
    }

    if (parsed_url.path.empty()) {
        parsed_url.path = "/";
    }

    SRV_INF("proxying %s request to %s://%s:%i%s\n", method.c_str(), parsed_url.scheme.c_str(), common_http_format_host(parsed_url.host).c_str(), parsed_url.port, parsed_url.path.c_str());

    std::map<std::string, std::string> headers;
    const std::string proxy_header_prefix = "x-llama-server-proxy-header-";
    for (const auto & entry : req.headers) {
        const std::string lowered_key = proxy_header_to_lower(entry.first);
        if (!string_starts_with(lowered_key, proxy_header_prefix)) {
            continue;
        }

        const std::string new_key = entry.first.substr(proxy_header_prefix.size());
        if (new_key.empty()) {
            continue;
        }

        headers[new_key] = entry.second;
    }

    return std::make_unique<server_http_proxy>(
            method,
            parsed_url.scheme,
            parsed_url.host,
            parsed_url.port,
            parsed_url.path,
            headers,
            req.body,
            req.files,
            req.should_stop,
            600,
            600);
}

static server_http_context::handler_t proxy_handler_post(std::vector<std::string> allowlist) {
    return [allowlist = std::move(allowlist)](const server_http_req & req) -> server_http_res_ptr {
        return proxy_request(req, "POST", allowlist);
    };
}

static server_http_context::handler_t proxy_handler_get(std::vector<std::string> allowlist) {
    return [allowlist = std::move(allowlist)](const server_http_req & req) -> server_http_res_ptr {
        return proxy_request(req, "GET", allowlist);
    };
}
