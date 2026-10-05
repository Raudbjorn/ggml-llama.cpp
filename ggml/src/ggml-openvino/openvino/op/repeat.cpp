#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/tile.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    auto input = process_view_input_new(context, 0);

    // to_shape() throws on a dynamic dimension, which the token axis is in dynamic or
    // stateful mode. Read the PartialShape instead and treat an unknown axis as a repeat
    // of 1, the only factor that is always valid: a repeat is an integer expansion, so
    // nothing can be tiled along an axis whose extent is not known yet.
    const auto input_shape  = context.get_input_shape(0);
    const auto output_shape = context.get_output_shape();

    FRONT_END_OP_CONVERSION_CHECK(input_shape.rank().is_static() && output_shape.rank().is_static(),
                                  "REPEAT requires a static rank on both input and output");

    std::vector<int64_t> repeats(4, 1);
    for (size_t axis = 0; axis < 4; ++axis) {
        if (!input_shape[axis].is_static() || !output_shape[axis].is_static()) {
            continue;
        }

        const int64_t input_dim  = input_shape[axis].get_length();
        const int64_t output_dim = output_shape[axis].get_length();

        FRONT_END_OP_CONVERSION_CHECK(input_dim > 0 && output_dim > 0 && output_dim % input_dim == 0,
                                      "REPEAT input shape cannot tile to match the output shape on axis ", axis);

        repeats[axis] = output_dim / input_dim;
    }

    auto repeats_node = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(input, repeats_node);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
