from typing import Any, Dict, List
import torch
from auto_diff import *


class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self,
        node_A: Node,
        node_B: Node,
        normalized_shape: List[int],
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        return torch.layer_norm(
            input_values[0] @ input_values[1],
            normalized_shape=node.attrs["normalized_shape"],
            eps=node.attrs["eps"]
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # ==========================================
        # compute gradient with respect to layernorm
        # ==========================================
        nd = len(node.attrs['normalized_shape'])
        dim = tuple(range(-nd, 0)) if nd > 0 else None
        eps = node.attrs['eps']

        x_hat = node  # use node output to compute std directly
        mean_x_hat_square = mean(power(x_hat, 2), dim=dim, keepdim=True)
        one_minus = sub(ones_like(mean_x_hat_square), mean_x_hat_square)
        # std = sqrt(eps / (1 - mean(x_hat²)))
        std = power(mul_by_const(power(one_minus, -1), eps), 0.5)

        # compute terms concerning gradient
        g_m = mean(output_grad, dim=dim, keepdim=True)
        g_hat_m = mean(mul(output_grad, x_hat), dim=dim, keepdim=True)
        # use sub() instead of - operator to create fewer nodes
        gradient = div(sub(sub(output_grad, g_m), mul(x_hat, g_hat_m)), std)

        # =======================================
        # combine gradient with respect to matmul
        # =======================================
        return [
            matmul(gradient, transpose(node.inputs[1], -1, -2)),
            matmul(transpose(node.inputs[0], -1, -2), gradient)
        ]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self,
        node_A: Node,
        node_B: Node,
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        return torch.softmax(input_values[0] @ input_values[1], dim=node.attrs["dim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        y = node
        # ========================================
        # compute gradient with respect to softmax
        # ========================================
        sum_y_grad = sum_op(mul(y, output_grad),
                            dim=node.attrs["dim"], keepdim=True)
        gradient = mul(y, sub(output_grad, sum_y_grad))

        # =======================================
        # combine gradient with respect to matmul
        # =======================================
        return [
            matmul(gradient, transpose(node.inputs[1], -1, -2)),
            matmul(transpose(node.inputs[0], -1, -2), gradient)
        ]


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
