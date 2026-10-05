#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
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

    // The decoder builds both shapes from the captured ggml extents, so they are static
    // rank 4 in every mode; the runtime-sized axis is reported by get_op_dynamic_dim().
    const auto input_shape  = context.get_input_shape(0).to_shape();
    const auto output_shape = context.get_output_shape().to_shape();

    FRONT_END_OP_CONVERSION_CHECK(input_shape.size() == 4 && output_shape.size() == 4,
                                  "REPEAT expects rank-4 shapes, got ", input_shape, " and ", output_shape);

    std::vector<int64_t> repeats(4, 1);
    for (size_t axis = 0; axis < 4; ++axis) {
        const int64_t input_dim  = input_shape[axis];
        const int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(input_dim > 0 && output_dim > 0 && output_dim % input_dim == 0,
                                      "REPEAT input shape ", input_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / input_dim;
    }

    ov::Output<ov::Node> repeats_node = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);

    // In a dynamic model the extent captured on the dynamic axis is stale. REPEAT keeps
    // src[0]'s dynamic dim, so a factor of 1 there tracks the input at any extent and stays
    // constant. Any other factor means the template is fixed on that axis (ggml_repeat keeps
    // no reference to it, only its extent), so divide that extent by the runtime input extent.
    const int32_t dynamic_dim = context.get_op_dynamic_dim();
    if (!context.is_static() && dynamic_dim >= 0 && dynamic_dim < 4) {
        const size_t dynamic_axis = 3 - dynamic_dim;  // OV order reverses ggml order
        if (repeats[dynamic_axis] != 1) {
            auto input_extent = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dynamic_axis}),
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            auto factor = std::make_shared<ov::op::v1::Divide>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) output_shape[dynamic_axis]}),
                input_extent);

            ov::OutputVector parts;
            for (size_t axis = 0; axis < 4; ++axis) {
                if (axis == dynamic_axis) {
                    parts.push_back(factor);
                } else {
                    parts.push_back(ov::op::v0::Constant::create(ov::element::i64, {1}, {repeats[axis]}));
                }
            }
            repeats_node = std::make_shared<ov::op::v0::Concat>(parts, 0);
        }
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(input, repeats_node);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
