# Automatic Differentiation and Tiny Transformer

## **Overview**

This project implements a **complete automatic differentiation (autodiff) framework from scratch**, along with a single-layer Transformer model built entirely on this custom framework. The primary goal is to provide a deep understanding of how modern deep learning frameworks like PyTorch and TensorFlow compute gradients through computational graphs, without relying on any pre-built autodiff libraries. By building these systems from the ground up, we gain insight into the mathematical foundations that power modern neural network training.

The project consists of three core modules that work together to demonstrate the complete pipeline of deep learning computation. The foundational module `auto_diff.py` implements a node-based computational graph system where each `Node` represents an operation (such as addition, multiplication, or matrix multiplication), and gradients can be automatically computed via **reverse-mode differentiation**. Building on this foundation, `transformer.py` demonstrates a practical application by constructing a complete Transformer encoder layer—including self-attention, feed-forward networks, and layer normalization—and training it on the MNIST dataset for digit classification. Finally, `fused_ops.py` introduces the concept of **operator fusion**, combining multiple operations (like matrix multiplication followed by softmax) into single computational nodes to reduce overhead and improve both memory efficiency and computational speed.

The design philosophy follows the **TensorFlow v1 paradigm**: we first define the computational graph symbolically using `Node` objects, then execute the graph by feeding concrete tensor values through an `Evaluator`. This static graph approach contrasts with PyTorch's eager execution model where operations are computed immediately. The advantage of our approach is that the entire computation structure is known before execution, enabling optimizations like common subexpression elimination and operator fusion. Through this implementation, you will understand not only how gradients flow backwards through complex neural network architectures, but also why certain operations (like softmax and layer normalization) require careful mathematical treatment during backpropagation.

---

## **auto_diff.py**

The `auto_diff.py` module serves as the heart of this project, implementing a complete automatic differentiation system based on computational graphs. At its core, the module defines two fundamental abstractions: the `Node` class representing computational graph vertices, and the `Op` class representing the operations performed at each vertex. Together, these abstractions enable the construction of arbitrary mathematical expressions that can be both evaluated forward to compute results and differentiated backward to compute gradients.

### Node Class and Operator Overloading

The `Node` class represents a single computation in the graph and maintains four essential fields: `inputs` stores references to the input nodes that feed into this computation, `op` stores the operation this node performs, `attrs` provides a dictionary for additional attributes like constant values, and `name` facilitates debugging. Through Python's operator overloading mechanism (`__add__`, `__mul__`, `__truediv__`, etc.), nodes can be combined using natural mathematical syntax—writing `x * y + z` automatically constructs a graph with multiplication and addition nodes, making graph construction feel as intuitive as writing mathematical equations while maintaining full control over the underlying computation structure.

```python
class Node:
    inputs: List["Node"]    # Input nodes to this computation
    op: "Op"               # The operation this node performs
    attrs: Dict[str, Any]  # Additional attributes (e.g., constants)
    name: str              # Name for debugging

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)
```

The operator overloading implementation demonstrates an important design pattern: when the operand is another `Node`, we create a binary operation node (like `AddOp`), but when the operand is a scalar constant, we create a unary operation with the constant stored in `attrs`. This distinction is crucial because constant operations have simpler gradients—adding a constant has gradient 1 with respect to the input, while multiplying by a constant has gradient equal to the constant itself.

### The Op Interface and Fundamental Operations

The `Op` class defines the interface that all operations must implement. Each operation provides three essential methods: `__call__` creates new nodes with the operation, `compute` performs the actual forward computation given input tensor values, and `gradient` constructs the backward graph nodes for gradient computation. This separation of concerns allows the same operation definition to be used for both forward evaluation and backward differentiation, embodying the mathematical duality between function evaluation and differentiation.

For arithmetic operations like multiplication, the gradient computation follows the **product rule** from calculus: given `f(a,b) = a*b`, we have `∂f/∂a = b` and `∂f/∂b = a`. In the context of automatic differentiation, we multiply these partial derivatives by the incoming gradient (the chain rule), resulting in the implementation where the partial adjoint with respect to the first input is the second input multiplied by the output gradient, and vice versa. Notice how the `gradient` method returns `Node` objects rather than tensor values—it constructs the backward computation graph symbolically, deferring actual computation to evaluation time.

```python
class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # Product rule: d(a*b)/da = b, d(a*b)/db = a
        return [mul(output_grad, node.inputs[1]), mul(output_grad, node.inputs[0])]
```

### Matrix Multiplication and Its Gradient

The `MatMulOp` is particularly important for neural networks, as it forms the backbone of all linear transformations in layers. The gradient computation for matrix multiplication requires careful handling of tensor dimensions and is often a source of confusion for those learning deep learning mathematics. Given the forward pass `C = A @ B` where `A` has shape `(m, n)` and `B` has shape `(n, p)`, the gradient with respect to `A` is `output_grad @ B^T`, and the gradient with respect to `B` is `A^T @ output_grad`. This relationship arises from the chain rule applied to matrix calculus—the transposition ensures that the dimensions align correctly for the matrix multiplications in the backward pass.

```python
class MatMulOp(Op):
    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        return input_values[0] @ input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        A, B = node.inputs
        return [
            matmul(output_grad, transpose(B, -1, -2)),  # dL/dA = dL/dC @ B^T
            matmul(transpose(A, -1, -2), output_grad)   # dL/dB = A^T @ dL/dC
        ]
```

To understand why these gradient formulas hold, consider the scalar-level chain rule. For a single element `C[i,j] = Σ_k A[i,k] * B[k,j]`, the partial derivative `∂C[i,j]/∂A[i,k] = B[k,j]`. When we aggregate these partials into matrix form and account for the incoming gradient `∂L/∂C`, we obtain exactly the matrix multiplication `∂L/∂A = ∂L/∂C @ B^T`. The implementation uses `transpose(B, -1, -2)` with negative indices to handle batched matrix multiplication correctly, swapping the last two dimensions regardless of the total number of dimensions.

### Softmax Gradient: A Non-Trivial Derivation

The `SoftmaxOp` presents one of the more challenging gradient computations in the module because **each output depends on all inputs** through the normalization denominator. For softmax applied along dimension `d`, the output is `y_i = exp(x_i) / Σ_j exp(x_j)`. When computing the Jacobian, we find that `∂y_i/∂x_j = y_i(δ_ij - y_j)` where `δ_ij` is the Kronecker delta. Applying the chain rule with an incoming gradient `g = ∂L/∂y`, we get `∂L/∂x_i = Σ_j g_j * y_j * (δ_ij - y_i) = y_i * g_i - y_i * Σ_j y_j * g_j = y_i * (g_i - Σ_j y_j * g_j)`.

```python
class SoftmaxOp(Op):
    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        return torch.softmax(input_values[0], dim=node.attrs["dim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # y = softmax(x), here node represents y
        # dL/dx = y * (dL/dy - sum(y * dL/dy))
        sum_y_grad = sum_op(mul(node, output_grad), dim=node.attrs["dim"], keepdim=True)
        return [mul(node, output_grad - sum_y_grad)]
```

The implementation is elegant: `node` represents the softmax output `y` (which is stored implicitly as part of the forward graph), `output_grad` represents `∂L/∂y`, and the term `sum(y * ∂L/∂y)` computes the weighted sum that appears in the gradient formula. The use of `keepdim=True` ensures proper broadcasting when subtracting from `output_grad`. This formulation avoids explicitly constructing the full Jacobian matrix, which would be quadratic in the sequence length.

### LayerNorm Gradient: Handling Statistical Dependencies

The `LayerNormOp` gradient is among the most complex in the module due to the **interdependence between the input and the computed statistics**. Layer normalization transforms the input as `x_hat = (x - μ) / σ` where `μ = mean(x)` and `σ = sqrt(var(x) + eps)`. The gradient must account for how each input element affects the normalized output both directly and indirectly through the mean and variance computations. When we differentiate through the mean and variance, we get additional terms that modify the simple `1/σ` gradient.

```python
class LayerNormOp(Op):
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = node.inputs[0]
        nd = len(node.attrs['normalized_shape'])
        dim = tuple(range(-nd, 0)) if nd > 0 else None
        
        # Compute forward statistics
        m = mean(x, dim=dim, keepdim=True)
        v = mean(power(x - m, 2), dim=dim, keepdim=True)
        std = power(add_by_const(v, node.attrs['eps']), 0.5)
        x_hat = div(x - m, std)
        
        # Gradient terms
        g_m = mean(output_grad, dim=dim, keepdim=True)
        g_hat_m = mean(mul(output_grad, x_hat), dim=dim, keepdim=True)
        
        return [div(output_grad - g_m - mul(x_hat, g_hat_m), std)]
```

The gradient formula `(g - mean(g) - x_hat * mean(g * x_hat)) / σ` has three components with clear interpretations. The first term `g/σ` is the direct gradient through the division by standard deviation. The second term `-mean(g)/σ` corrects for how each input affects the mean (the mean depends on all inputs equally, so the correction is uniform). The third term `-x_hat * mean(g * x_hat) / σ` corrects for how each input affects the variance—inputs far from the mean contribute more to variance, and this dependence is captured by the correlation between `g` and `x_hat`.

### Broadcasting and Gradient Aggregation

The `BroadcastOp` handles the expansion of tensors to larger shapes, which is essential for element-wise operations between tensors of different dimensions. The key insight for the backward pass is that **gradients must be summed along the dimensions that were broadcast**. When a tensor is broadcast from shape `(1, n)` to `(m, n)`, the single row is conceptually replicated `m` times. During backpropagation, the gradients from all `m` uses must be aggregated back to the original shape, which is accomplished through summation.

```python
class BroadcastOp(Op):
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]
        
        # Find dimensions that were broadcast (size 1 expanded to larger)
        dims_to_sum = []
        for i, (in_size, out_size) in enumerate(zip(input_shape[::-1], output_shape[::-1])):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)
        
        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)
        
        # Handle added dimensions at the front
        if len(output_shape) > len(input_shape):
            grad = sum_op(grad, dim=list(range(len(output_shape) - len(input_shape))), keepdim=False)
        
        return [grad]
```

The implementation compares shapes from the right (innermost dimensions first) to identify which dimensions were expanded. This right-alignment matches NumPy/PyTorch broadcasting semantics where shapes are compared starting from trailing dimensions. The algorithm handles both cases: dimensions of size 1 that were expanded to a larger size (requiring sum with `keepdim=True` to preserve the dimension), and completely new dimensions that were prepended (requiring sum with `keepdim=False` to remove them).

### Topological Sort and Reverse-Mode Autodiff

The `gradients` function implements the core **reverse-mode automatic differentiation** algorithm. It first discovers all nodes reachable from the output using depth-first search, then traverses them in reverse topological order to propagate gradients backward. The topological sort, implemented using Kahn's algorithm with in-degree tracking, ensures that each node is processed only after all nodes that depend on it have been processed—this is the reverse of the forward evaluation order.

```python
def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    # DFS to find all reachable nodes
    visited, all_nodes = set(), []
    stack = [output_node]
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node)
        all_nodes.append(node)
        stack.extend(node.inputs)

    # Initialize output gradient to 1 (implicit dL/dL = 1)
    node_to_grad = {output_node: ones_like(output_node)}

    # Reverse topological traversal for backpropagation
    for node in reversed(topological_sort(all_nodes)):
        if node not in node_to_grad: continue
        if node.op is placeholder: continue
        
        input_grads = node.op.gradient(node, node_to_grad[node])
        
        # Accumulate gradients for nodes with multiple consumers
        for input_node, input_grad in zip(node.inputs, input_grads):
            if input_node in node_to_grad:
                node_to_grad[input_node] += input_grad  # Sum rule
            else:
                node_to_grad[input_node] = input_grad
    
    return [node_to_grad.get(n, zeros_like(n)) for n in nodes]
```

The gradient accumulation step (using `+=`) is crucial for correctness when a node has multiple consumers. If a variable `x` is used in both `y = x + 1` and `z = x * 2`, the gradient `∂L/∂x` must be the sum of `∂L/∂y * ∂y/∂x + ∂L/∂z * ∂z/∂x`. This is the multivariate chain rule in action, and the topological order traversal ensures that when we process `x`, we have already accumulated gradients from all of its consumers.

---

## **transformer.py**

The `transformer.py` module demonstrates a practical application of the autodiff framework by implementing a complete single-layer Transformer encoder and training it on the MNIST handwritten digit classification task. The implementation treats each 28×28 MNIST image as a sequence of 28 tokens (rows), where each token is a 28-dimensional vector (pixel values in that row). This approach showcases how the self-attention mechanism can capture spatial relationships between different rows of an image, providing an interesting alternative to the traditional convolutional approach for image classification.

### Linear Transformation Layer

The `linear` function implements the fundamental building block of neural networks: an affine transformation `output = x @ W + b`. While conceptually simple, this operation combines matrix multiplication for weight application with addition for bias, demonstrating how complex operations are built from primitive autodiff operations. The broadcasting behavior of addition automatically handles the bias vector being added to each sample in the batch and each position in the sequence—a `(d,)` shaped bias broadcasts correctly to a `(batch, seq_len, d)` shaped matmul output.

```python
def linear(x: ad.Node, W: ad.Node, b: ad.Node) -> ad.Node:
    """Linear transformation: output = x @ W + b"""
    return ad.add(ad.matmul(x, W), b)
```

### Self-Attention Mechanism

The `attention` function implements the core **scaled dot-product self-attention** mechanism that gives Transformers their remarkable ability to model long-range dependencies. The implementation first projects the input `x` into queries (`Q`), keys (`K`), and values (`V`) through learned weight matrices. Attention scores are computed as `Q @ K^T`, scaled by `1/sqrt(d_k)` to prevent gradient vanishing in softmax, normalized with softmax to produce attention weights, and finally used to compute a weighted sum of values.

```python
def attention(x: ad.Node, W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node, model_dim: int) -> ad.Node:
    # Project input to Q, K, V
    q = ad.matmul(x, W_Q)
    k = ad.matmul(x, W_K)
    v = ad.matmul(x, W_V)

    # Scaled dot-product attention
    score = ad.matmul(q, ad.transpose(k, 1, 2))          # Q @ K^T: (batch, seq, seq)
    norm_score = ad.div_by_const(score, (model_dim ** 0.5))  # Scale by sqrt(d_k)
    attn_weights = ad.softmax(norm_score, dim=-1)        # Softmax over key positions
    
    return ad.matmul(attn_weights, v)  # Weighted sum of values
```

The scaling factor `1/sqrt(model_dim)` is crucial for training stability and deserves special attention. Without scaling, the dot products `q_i · k_j` grow in magnitude proportionally to the dimension `d_k`—if each element of `q` and `k` has variance 1, their dot product has variance `d_k`. Large dot products push the softmax into saturation regions where gradients become extremely small, a phenomenon known as **attention score explosion**. The `1/sqrt(d_k)` scaling normalizes the variance of dot products back to approximately 1, keeping the softmax in its sensitive region throughout training.

### Complete Transformer Encoder Layer

The `transformer` function assembles a complete encoder layer by combining self-attention with a feed-forward network. The architecture follows the pattern: **attention → output projection → layer normalization → FFN → layer normalization**. This implementation omits residual connections for pedagogical clarity, though production Transformers use them to facilitate gradient flow through deep networks and enable the model to learn identity mappings easily.

```python
def transformer(X: ad.Node, nodes: List[ad.Node], model_dim: int, num_classes: int, eps: float) -> ad.Node:
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # Self-attention sub-layer
    att_output = attention(X, W_Q, W_K, W_V, model_dim)
    att_output = ad.matmul(att_output, W_O)
    norm_output = ad.layernorm(att_output, [model_dim], eps=eps)

    # Feed-forward sub-layer
    ffn_hidden = ad.relu(linear(norm_output, W_1, b_1))
    ffn_hidden = linear(ffn_hidden, W_2, b_2)
    norm_output = ad.layernorm(ffn_hidden, [num_classes], eps=eps)

    return norm_output
```

The feed-forward network (FFN) consists of two linear transformations with a ReLU activation in between. In full Transformers, the hidden dimension is typically 4× the model dimension, creating a "bottleneck" structure that first expands the representation then compresses it back. The layer normalization applied after each sub-layer stabilizes training by ensuring that the inputs to subsequent layers have consistent statistics, which is particularly important when training without residual connections.

### Cross-Entropy Loss with Hierarchical Summation

The `softmax_loss` function computes the average cross-entropy loss between predicted logits and one-hot encoded ground truth labels. For a 3D tensor where shapes are `(batch, seq_len, num_classes)`, we must carefully aggregate the loss across all dimensions while maintaining numerical correctness. The implementation first applies softmax to convert logits to probabilities, computes log probabilities, multiplies element-wise with one-hot labels to select the log probability of the correct class, and sums hierarchically.

```python
def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    log_probs = ad.log(ad.softmax(Z, dim=-1))
    cross_entropy = ad.mul(y_one_hot, log_probs)

    # Hierarchical summation for 3D tensors
    total_loss = ad.sum_op(cross_entropy, dim=2, keepdim=True)  # sum over classes
    total_loss = ad.sum_op(total_loss, dim=1, keepdim=True)     # sum over sequence
    total_loss = ad.sum_op(total_loss, dim=0, keepdim=True)     # sum over batch

    return ad.mul_by_const(ad.div_by_const(total_loss, float(batch_size)), -1.0)
```

The hierarchical summation approach (sum over classes → sum over sequence → sum over batch) is used instead of a single sum because our `sum_op` implementation handles multi-dimensional summation more reliably when done incrementally. The explicit `batch_size` parameter is needed because our autodiff framework requires knowing the normalization factor at graph construction time, unlike PyTorch which can infer it dynamically from tensor shapes at runtime.

### Training Loop and Model Evaluation

The training pipeline demonstrates the complete workflow of using our autodiff framework. The `train_model` function creates `Variable` nodes for all inputs and parameters, constructs the forward graph through `transformer` and `softmax_loss`, computes gradient nodes using `ad.gradients`, and creates evaluators for both training (forward + backward) and testing (forward only). The SGD update rule `param = param - lr * grad` is applied directly to the weight tensors, demonstrating the clean separation between graph construction and graph execution.

After training for 20 epochs with batch size 50 and learning rate 0.02, the model typically achieves around **50% test accuracy** on MNIST. While modest by modern standards, this result is remarkable given the simplicity of the architecture (single layer, no residual connections, no multi-head attention, no positional encoding) and the fact that the entire training pipeline runs on a custom autodiff implementation.

---

## **fused_ops.py**

The `fused_ops.py` module introduces **operator fusion**, a critical optimization technique in deep learning systems that combines multiple sequential operations into a single computational kernel. In standard execution, each operation reads its inputs from memory, performs computation, and writes results back to memory. For a sequence like matrix multiplication followed by layer normalization, the intermediate result must be materialized in GPU global memory—a slow operation that can dominate runtime in memory-bandwidth-limited scenarios. Fused operations eliminate these intermediate memory accesses by keeping data in faster cache or registers throughout the combined computation.

### MatMul + LayerNorm Fusion

The `MatMulLayerNormOp` combines matrix multiplication with layer normalization, a pattern that appears after every attention and feed-forward sub-layer in Transformers. The forward computation is straightforward—we simply chain the PyTorch operations. The gradient computation, however, is more sophisticated and demonstrates an **elegant optimization**: instead of recomputing `x = A @ B` in the backward pass to compute the layer normalization gradient, we can derive the standard deviation directly from the normalized output `x_hat` using the mathematical identity `std = sqrt(eps / (1 - mean(x_hat²)))`.

```python
class MatMulLayerNormOp(Op):
    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        return torch.layer_norm(
            input_values[0] @ input_values[1],
            normalized_shape=node.attrs["normalized_shape"],
            eps=node.attrs["eps"]
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        nd = len(node.attrs['normalized_shape'])
        dim = tuple(range(-nd, 0)) if nd > 0 else None
        eps = node.attrs['eps']

        # KEY OPTIMIZATION: Derive std from x_hat (=node) without recomputing x = A @ B
        # Mathematical insight: mean(x_hat²) = var / (var + eps)
        # Solving: std = sqrt(eps / (1 - mean(x_hat²)))
        x_hat = node
        mean_x_hat_sq = mean(power(x_hat, 2), dim=dim, keepdim=True)
        one_minus = sub(ones_like(mean_x_hat_sq), mean_x_hat_sq)
        std = power(mul_by_const(power(one_minus, -1), eps), 0.5)

        # LayerNorm gradient using derived std
        g_m = mean(output_grad, dim=dim, keepdim=True)
        g_hat_m = mean(mul(output_grad, x_hat), dim=dim, keepdim=True)
        gradient = div(sub(sub(output_grad, g_m), mul(x_hat, g_hat_m)), std)

        # Chain with MatMul gradient
        return [
            matmul(gradient, transpose(node.inputs[1], -1, -2)),
            matmul(transpose(node.inputs[0], -1, -2), gradient)
        ]
```

The mathematical derivation of this optimization deserves explanation. For layer-normalized output `x_hat = (x - μ) / σ`, we have `mean(x_hat) = 0` and `var(x_hat) = mean(x_hat²) = var(x) / σ² = var(x) / (var(x) + eps)`. Solving for `σ`, we get `σ² = eps / (1 - mean(x_hat²))`. This allows us to compute the standard deviation needed for the gradient **without storing or recomputing the pre-normalization values**, reducing both memory usage and computation in the backward pass.

### MatMul + Softmax Fusion

The `MatMulSoftmaxOp` fuses matrix multiplication with softmax, directly corresponding to the attention score computation in Transformers where we compute `softmax(Q @ K^T)`. This fusion is particularly valuable because the intermediate `Q @ K^T` matrix can be very large—for sequence length `L`, this matrix has `L × L` elements per batch and head. Avoiding the memory round-trip for this intermediate result significantly reduces memory pressure, especially for long sequences.

```python
class MatMulSoftmaxOp(Op):
    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        return torch.softmax(input_values[0] @ input_values[1], dim=node.attrs["dim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        y = node  # Softmax output is available as the node itself
        
        # Softmax gradient: y * (g - sum(y * g))
        sum_y_grad = sum_op(mul(y, output_grad), dim=node.attrs["dim"], keepdim=True)
        gradient = mul(y, sub(output_grad, sum_y_grad))

        # Chain with MatMul gradient
        return [
            matmul(gradient, transpose(node.inputs[1], -1, -2)),
            matmul(transpose(node.inputs[0], -1, -2), gradient)
        ]
```

Unlike the MatMul+LayerNorm case, the softmax gradient computation can use the softmax output directly (stored as `node`) without any clever mathematical reformulation. This is because the softmax gradient formula `y * (g - Σ(y*g))` only depends on the softmax output `y`, not the pre-softmax logits. The gradient then flows through the matmul backward pass using the standard transposition rules.

### Performance Implications and System Context

While this module implements fusion at the Python level (which doesn't provide the full performance benefits of true kernel fusion), it illustrates the conceptual foundation that underlies optimizations in production frameworks like **TensorRT**, **XLA**, and **Triton**. In real deployments with custom CUDA kernels, fused operations can provide 2-10× speedups by eliminating memory bottlenecks. The performance gain comes from three sources: reduced memory bandwidth consumption (the dominant cost in many operations), reduced kernel launch overhead (each kernel launch has microsecond-level latency), and improved cache utilization (intermediate results stay in L1/L2 cache or registers).

The fused operations also serve as a conceptual bridge to understanding more advanced optimization techniques. Modern compilers like XLA automatically discover fusion opportunities by analyzing the computational graph—the same graph structure we build explicitly in this project. Understanding fusion at this level prepares you for working with auto-tuning systems, custom Triton kernels, and the increasingly important field of ML compiler optimization.

---

## **Getting Started**

### Running Tests

```bash
# Test individual operator forward/backward passes
pytest tests/test_auto_diff_node_forward.py
pytest tests/test_auto_diff_node_backward.py

# Test full graph evaluation
pytest tests/test_auto_diff_graph_forward.py
pytest tests/test_auto_diff_graph_backward.py

# Test fused operations
pytest tests/test_fused_ops.py
python tests/test_fused_ops_perf.py  # Performance comparison
```

### Training the Transformer

```bash
python transformer.py
```

This will train a single-layer Transformer on MNIST for 20 epochs. Expected final test accuracy is approximately **50%** (this simplified architecture intentionally omits residual connections and multi-head attention for pedagogical clarity).

---

## **Key Concepts Summary**

| Concept | Location | Description |
|---------|----------|-------------|
| Computational Graph | `auto_diff.py` | Node-based representation of mathematical expressions |
| Reverse-Mode Autodiff | `gradients()` | Backpropagation via reverse topological traversal |
| Operator Abstraction | `Op` class | Unified interface for `compute`/`gradient` methods |
| Self-Attention | `transformer.py` | Scaled dot-product attention mechanism |
| Operator Fusion | `fused_ops.py` | Combining operations to reduce memory overhead |
