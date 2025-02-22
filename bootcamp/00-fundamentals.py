import torch

# DEF: Tensor - A method of storing multidimensional data (e.g., a matrix is a 2D tensor).

SCALAR = torch.tensor(7)
SCALAR_DIM = SCALAR.ndim
SCALAR_VAL = SCALAR.item()
print(f"Create a scalar: {SCALAR}")
print(f"Dimensions of the scalar: {SCALAR_DIM}")
print(f"Value of the scalar: {SCALAR_VAL}")
print()

VECTOR = torch.tensor([7, 7])
VECTOR_DIM = VECTOR.ndim
VECTOR_SHAPE = VECTOR.shape
print(f"Create a vector: {VECTOR}")
print(f"Dimensions of the vector: {VECTOR_DIM}")
print(f"Shape of the vector: {VECTOR_SHAPE}")
print()

MATRIX = torch.tensor([[7, 8], [9, 12]])
MATRIX_DIM = MATRIX.ndim
MATRIX_SHAPE = MATRIX.shape
print(f"Create a matrix: {MATRIX}")
print(f"Dimensions of the matrix: {MATRIX_DIM}")
print(f"Shape of the matrix: {MATRIX_SHAPE}")
print()

TENSOR = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
TENSOR_DIM = TENSOR.ndim
TENSOR_SHAPE = TENSOR.shape
print(f"Create a tensor: {TENSOR}")
print(f"Dimensions of the tensor: {TENSOR_DIM}")
print(f"Shape of the tensor: {TENSOR_SHAPE}")
print(f"The 0th element of the tensor: {TENSOR[0]}")
print()

# can I create an uneven tensor? (NO)
# print(torch.tensor([[[1,2,3],[4,5],[6]]]))

# Random tensors - Important for initialisation
random_tensor = torch.rand(3, 4)
print(f"Random tensor: {random_tensor}")

# Create random tensor to similar size as image
random_image_tensor = torch.rand(size=(3, 224, 224))  # height, width, RGB
print(random_image_tensor)

# NOTE: size and shape return/work the same

# Create a tensor of zeros

zeros = torch.zeros(3, 4)
print(zeros)

ones = torch.ones(3, 4)
print(ones)

# dtype is 'default type'
print(f"Default data type: {ones.dtype}")

# Create a range of tensors
tensor_range = torch.arange(start=0, end=1000, step=77)
print(tensor_range)

# Create tensors 'like' another (e.g., same shape)
tensor_like = torch.zeros_like(input=tensor_range)
print(tensor_like)
print()

float_32_tensor = torch.tensor(
    [3.0, 6.0],
    dtype=None,  # data type (32 bit float, etc.)
    device=None,  # CPU/GPU/etc. - tensors need to exist on the same device in order to interact
    requires_grad=False,  # whether or not to track the gradient with operations on the tensor
)
print(f"Float 32 tensor: {float_32_tensor.dtype}")
# using smaller data types (e.g., 16-bit) will make computations faster, at the cost of precision
float_16_tensor = float_32_tensor.type(torch.float16)
print(f"Float16 tensor: {float_16_tensor.dtype}")
print(
    f"When multiplying float16 by float32, the result is 32: {(float_16_tensor * float_32_tensor).dtype}"
)
print()

# Manipulating tensors
tensor_to_maniuplate = torch.tensor([1, 2, 3])
print(f"Add 10 to {tensor_to_maniuplate}: {tensor_to_maniuplate + 10}")
print(f"Multiply {tensor_to_maniuplate} by 10: {tensor_to_maniuplate * 10}")
# ...etc...
print()

# Matrix multiplication
print(
    tensor_to_maniuplate,
    "*",
    tensor_to_maniuplate,
    " is ",
    tensor_to_maniuplate * tensor_to_maniuplate,
)
print(
    tensor_to_maniuplate,
    " dot ",
    tensor_to_maniuplate,
    " is ",
    torch.matmul(tensor_to_maniuplate, tensor_to_maniuplate),
)
# Shape errors
# 1. Inner dimensions must match:
#   (row height must match column height)
#   - (3,2) @ (3,2) WON'T work
#   - (2,3) @ (3,2) WILL work
#   - (3,2) @ (2,3) WILL work
# 2. The resulting matrix has the shape of the outer dimensions:
#   (each row applied to each column)
#   - (2,3) @ (3,2) -> (2,2)
print()

# Transposing to fix shape issues
tensor_A = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor_B = torch.tensor([[7, 10], [8, 11], [9, 12]])
print(
    f"Cannot matmul {tensor_A} and {tensor_B} as their inner dimensions do not match: {tensor_A.shape, tensor_B.shape}"
)
# transpose switches the dimensions/axes of a given tensor
print(
    f"Transpose a matrix to switch their axes and fix the incompatibility: {tensor_A, tensor_B.T}"
)
print(f"So now they can be matrix multiplied: {torch.mm(tensor_A, tensor_B.T)}")
print(
    "NOTE: the same information is represented in these tensors, it has simply been rearranged in one of them."
)
print()


# Tensor aggregation
agg_tensor = torch.arange(0, 100, 10)
# find the min
print(f"Min of tensor {agg_tensor}: {agg_tensor.min()}")
# find the max
print(f"Max of tensor {agg_tensor}: {torch.max(agg_tensor)}")
# find the mean (note: mean can not be applied to float64 dtypes)
print(f"Mean of tensor {agg_tensor}: {torch.mean(agg_tensor.type(torch.float32))}")
# find the sum
print(f"Sum of tensor {agg_tensor}: {agg_tensor.sum()}")
# find positional max
print(f"Positional max of tensor {agg_tensor}: {agg_tensor.argmax()}")
# find positional min
print(f"Positional min of tensor {agg_tensor}: {agg_tensor.argmin()}")
print()


# Reshaping tensors
reshape_tensor = torch.arange(1, 10)
# add a dimension
print(f"Reshape tensor {reshape_tensor} to 1x9: {reshape_tensor.reshape(1, 9)}")
print(f"Reshape tensor {reshape_tensor} to 9x1: {reshape_tensor.reshape(9, 1)}")
# note total number of elements must remain the same for a reshape to work
print()

# Change the view (same as reshape but just 'looks' at an existing tensor)
view_tensor = torch.arange(1, 10)
print(f"View tensor {view_tensor} to 1x9: {view_tensor.view(1, 9)}")
# note: a view shares memory with the original tensor (is a reference)
print()

# Stack tensors
stack_tensor_A = torch.tensor([1, 2, 3])
stack_tensor_B = torch.tensor([4, 5, 6])
print(
    f"Stack tensors {stack_tensor_A} and {stack_tensor_B}: {torch.stack([stack_tensor_A, stack_tensor_B], dim=1)}"
)
print()

# Squeeze tensors (remove dimensions of size 1 - kinda like 'redundant' dimensions)
squeeze_tensor = torch.arange(0, 9).reshape(3, 1, 3)
print(f"Squeeze tensor {squeeze_tensor}: {squeeze_tensor.squeeze()}")
# Unsqueeze tensors (add dimensions of size 1)
unsqueeze_tensor = torch.arange(0, 9).reshape(3, 3)
print(f"Unsqueeze tensor {unsqueeze_tensor}: {unsqueeze_tensor.unsqueeze(dim=2)}")
print()

# Permute tensors (rearrange dimensions in a specified order)
permute_tensor = torch.rand(size=(224, 224, 3))
print(
    f"Permute tensor {permute_tensor.shape} by shfting axis 0->1, 1->2, 2->0: {permute_tensor.permute(1, 2, 0).shape}"
)
permute_tensor_small = torch.rand(size=(2, 2))
print(
    f"Permute tensor {permute_tensor_small} by swapping axis 0->1, 1->0: {permute_tensor_small.permute(1, 0)}"
)
print()

# Indexing tensors
index_tensor = torch.arange(1, 10).reshape(1, 3, 3)
print(f"Tensor to index: {index_tensor}")
print(f"index_tensor[0][0]: {index_tensor[0][0]}")
print(f"index_tensor[0][0][0]: {index_tensor[0][0][0]}")
print(f"index_tensor[0][2][2]: {index_tensor[0][2][2]}")
print(f"index_tensor[0][:]: {index_tensor[0][:]}")
print()

# PyTorch tensors and NumPy
# might start with np data and want a torch tensor
import numpy as np

np_array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(np_array)
print(f"Tensor from NumPy array: {tensor}")
# note np default dtype is float64, whereas torch default dtype is float32
tensor_for_np = torch.ones(7)
np_array = tensor_for_np.numpy()
print(f"NumPy array from tensor: {np_array}")
# note dtypes are preserved through both conversions
print()

# Reproducibility
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
print(f"Random tensor A: {random_tensor_A}")
print(f"Random tensor B: {random_tensor_B}")
print(f"Are they equal? {random_tensor_A == random_tensor_B}")
torch.manual_seed(42)
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(42)
random_tensor_D = torch.rand(3, 4)
print(f"Random tensor C: {random_tensor_C}")
print(f"Random tensor D: {random_tensor_D}")
print(f"Are they equal? {random_tensor_C == random_tensor_D}")
# note that the seed must be set before each randomness operation
print()

# Running on GPUs
print(f"Check for GPU access: {torch.cuda.is_available()}")

# setup device-agnostic code!
device = "cuda" if torch.cuda.is_available() else "cpu"

tensor_cpu = torch.rand(1, 3)
print(f"Tensor on CPU: {tensor_cpu, tensor_cpu.device}")
tensor_gpu = torch.rand(1, 3).to(device)
print(f"Tensor on GPU: {tensor_gpu, tensor_gpu.device}")
tensor_cpu_to_gpu = tensor_cpu.to(device)
print(f"Tensor moved to GPU: {tensor_cpu_to_gpu, tensor_cpu_to_gpu.device}")
print(
    f"NumPy only supports CPU operations, so tensors must be moved to CPU to convert to NumPy arrays: {tensor_cpu_to_gpu.cpu().numpy()}"
)
