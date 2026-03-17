import numpy as np
import torch
from torch.autograd import Function
from numba import jit, cuda
import math


@jit(nopython=True)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


@jit(nopython=True)
def my_min(x, gamma):
    min_x, argmax_x = my_max(-x, gamma)
    return - min_x, argmax_x


@jit(nopython=True)
def my_max_hessian_product(p, z, gamma):
    return (p * z - p * np.sum(p * z)) / gamma


@jit(nopython=True)
def my_min_hessian_product(p, z, gamma):
    return - my_max_hessian_product(p, z, gamma)


@jit(nopython=True)
def dtw_grad(theta, gamma):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = my_min(np.array([V[i, j - 1],
                                          V[i - 1, j - 1],
                                          V[i - 1, j]]), gamma)
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                      Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                      Q[i + 1, j, 2] * E[i + 1, j]

    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


@jit(nopython=True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    m = Z.shape[0]
    n = Z.shape[1]

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            V_dot[i, j] = Z[i - 1, j - 1] + \
                          Q[i, j, 0] * V_dot[i, j - 1] + \
                          Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                          Q[i, j, 2] * V_dot[i - 1, j]

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)
    E_dot = np.zeros((m + 2, n + 2))

    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                          Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                          Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                          Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                          Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                          Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]


class PathDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma):  # D.shape: [batch_size, N , N]
        batch_size, N, N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)

        grad_gpu = torch.zeros((batch_size, N, N)).to(device)
        Q_gpu = torch.zeros((batch_size, N + 2, N + 2, 3)).to(device)
        E_gpu = torch.zeros((batch_size, N + 2, N + 2)).to(device)

        for k in range(0, batch_size):  # loop over all D in the batch
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k, :, :], gamma)
            grad_gpu[k, :, :] = torch.FloatTensor(grad_cpu_k).to(device)
            Q_gpu[k, :, :, :] = torch.FloatTensor(Q_cpu_k).to(device)
            E_gpu[k, :, :] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu, D, Q_gpu, E_gpu, gamma_gpu)
        return torch.mean(grad_gpu, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()

        batch_size, N, N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N, N)).to(device)
        for k in range(0, batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k, :, :], Z, Q_cpu[k, :, :, :], E_cpu[k, :, :], gamma)
            Hessian[k:k + 1, :, :] = torch.FloatTensor(hess_k).to(device)

        return Hessian, None


# ============================================================================
# GPU CUDA Implementation
# ============================================================================

@cuda.jit(device=True)
def my_min_cuda(v0, v1, v2, gamma):
    """CUDA device function for computing soft minimum and probabilities"""
    # Compute soft minimum using log-sum-exp trick
    # Negate values for minimum (since we use max trick)
    val0 = -v0
    val1 = -v1
    val2 = -v2

    max_val = max(max(val0, val1), val2)

    exp0 = math.exp((val0 - max_val) / gamma)
    exp1 = math.exp((val1 - max_val) / gamma)
    exp2 = math.exp((val2 - max_val) / gamma)

    Z = exp0 + exp1 + exp2

    min_val = -(gamma * math.log(Z) + max_val)

    # Probabilities
    q0 = exp0 / Z
    q1 = exp1 / Z
    q2 = exp2 / Z

    return min_val, q0, q1, q2


@cuda.jit
def path_dtw_forward_cuda(D, gamma, N, n_passes, V, Q, E):
    """
    Forward pass computing the gradient (path matrix) using anti-diagonal parallelization
    Each block processes one sample in the batch
    """
    b = cuda.blockIdx.x  # batch index
    tid = cuda.threadIdx.x  # thread index

    I = tid

    # Forward pass: compute V and Q matrices
    for p in range(n_passes):
        J = max(0, min(p - tid, N - 1))
        i = I + 1
        j = J + 1

        # Only compute if on current anti-diagonal and within bounds
        if I + J == p and I < N and J < N:
            v0 = V[b, i, j - 1]
            v1 = V[b, i - 1, j - 1]
            v2 = V[b, i - 1, j]

            min_val, q0, q1, q2 = my_min_cuda(v0, v1, v2, gamma)

            V[b, i, j] = D[b, I, J] + min_val
            Q[b, i, j, 0] = q0
            Q[b, i, j, 1] = q1
            Q[b, i, j, 2] = q2

        cuda.syncthreads()

    # Backward pass: compute E matrix (gradient)
    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, N - 1))
        i = I + 1
        j = J + 1

        if I + J == rev_p and I < N and J < N:
            E[b, i, j] = (Q[b, i, j + 1, 0] * E[b, i, j + 1] +
                          Q[b, i + 1, j + 1, 1] * E[b, i + 1, j + 1] +
                          Q[b, i + 1, j, 2] * E[b, i + 1, j])

        cuda.syncthreads()


@cuda.jit
def path_dtw_backward_cuda(D, Z, Q, E, gamma, N, n_passes, V_dot, Q_dot, E_dot, Hessian):
    """
    Backward pass computing the Hessian product
    """
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    I = tid
    inv_gamma = 1.0 / gamma

    # Forward pass for V_dot and Q_dot
    for p in range(n_passes):
        J = max(0, min(p - tid, N - 1))
        i = I + 1
        j = J + 1

        if I + J == p and I < N and J < N:
            V_dot[b, i, j] = (Z[I, J] +
                              Q[b, i, j, 0] * V_dot[b, i, j - 1] +
                              Q[b, i, j, 1] * V_dot[b, i - 1, j - 1] +
                              Q[b, i, j, 2] * V_dot[b, i - 1, j])

            v0 = V_dot[b, i, j - 1]
            v1 = V_dot[b, i - 1, j - 1]
            v2 = V_dot[b, i - 1, j]

            # Hessian product for Q_dot
            q0 = Q[b, i, j, 0]
            q1 = Q[b, i, j, 1]
            q2 = Q[b, i, j, 2]

            sum_qv = q0 * v0 + q1 * v1 + q2 * v2

            Q_dot[b, i, j, 0] = -(q0 * v0 - q0 * sum_qv) * inv_gamma
            Q_dot[b, i, j, 1] = -(q1 * v1 - q1 * sum_qv) * inv_gamma
            Q_dot[b, i, j, 2] = -(q2 * v2 - q2 * sum_qv) * inv_gamma

        cuda.syncthreads()

    # Backward pass for E_dot
    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, N - 1))
        i = I + 1
        j = J + 1

        if I + J == rev_p and I < N and J < N:
            E_dot[b, i, j] = (Q_dot[b, i, j + 1, 0] * E[b, i, j + 1] +
                              Q[b, i, j + 1, 0] * E_dot[b, i, j + 1] +
                              Q_dot[b, i + 1, j + 1, 1] * E[b, i + 1, j + 1] +
                              Q[b, i + 1, j + 1, 1] * E_dot[b, i + 1, j + 1] +
                              Q_dot[b, i + 1, j, 2] * E[b, i + 1, j] +
                              Q[b, i + 1, j, 2] * E_dot[b, i + 1, j])

            # Store result
            Hessian[b, I, J] = E_dot[b, i, j]

        cuda.syncthreads()


class PathDTWBatchCUDA(Function):
    """GPU-accelerated version of PathDTWBatch using CUDA"""

    @staticmethod
    def forward(ctx, D, gamma):
        """
        Args:
            D: Distance matrix [batch_size, N, N]
            gamma: Smoothing parameter
        Returns:
            grad: Average gradient matrix [N, N]
        """
        batch_size, N, _ = D.shape
        device = D.device
        dtype = D.dtype

        # Check if we can use CUDA
        if not device.type == 'cuda':
            raise RuntimeError("PathDTWBatchCUDA requires CUDA device")

        if N > 1024:
            raise RuntimeError(f"Sequence length {N} exceeds CUDA block size limit (1024)")

        # Prepare arrays on GPU
        V = torch.ones((batch_size, N + 2, N + 2), device=device, dtype=dtype) * 1e10
        V[:, 0, 0] = 0

        Q = torch.zeros((batch_size, N + 2, N + 2, 3), device=device, dtype=dtype)

        E = torch.zeros((batch_size, N + 2, N + 2), device=device, dtype=dtype)
        E[:, N + 1, N + 1] = 1
        Q[:, N + 1, N + 1, :] = 1

        threads_per_block = N
        n_passes = 2 * N - 1

        # Launch CUDA kernel
        path_dtw_forward_cuda[batch_size, threads_per_block](
            cuda.as_cuda_array(D.detach()),
            gamma, N, n_passes,
            cuda.as_cuda_array(V),
            cuda.as_cuda_array(Q),
            cuda.as_cuda_array(E)
        )

        # Extract gradient (E matrix)
        grad = E[:, 1:N + 1, 1:N + 1]

        # Save for backward
        ctx.save_for_backward(D, Q, E, torch.tensor([gamma], device=device, dtype=dtype))
        ctx.N = N

        return torch.mean(grad, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: Gradient from upstream [N, N]
        Returns:
            Gradient w.r.t. D [batch_size, N, N]
        """
        D, Q, E, gamma_tensor = ctx.saved_tensors
        N = ctx.N
        batch_size = D.shape[0]
        device = D.device
        dtype = D.dtype

        gamma = gamma_tensor.item()
        Z = grad_output.detach()

        # Prepare arrays
        V_dot = torch.zeros((batch_size, N + 2, N + 2), device=device, dtype=dtype)
        Q_dot = torch.zeros((batch_size, N + 2, N + 2, 3), device=device, dtype=dtype)
        E_dot = torch.zeros((batch_size, N + 2, N + 2), device=device, dtype=dtype)
        Hessian = torch.zeros((batch_size, N, N), device=device, dtype=dtype)

        threads_per_block = N
        n_passes = 2 * N - 1

        # Launch CUDA kernel
        path_dtw_backward_cuda[batch_size, threads_per_block](
            cuda.as_cuda_array(D.detach()),
            cuda.as_cuda_array(Z),
            cuda.as_cuda_array(Q),
            cuda.as_cuda_array(E),
            gamma, N, n_passes,
            cuda.as_cuda_array(V_dot),
            cuda.as_cuda_array(Q_dot),
            cuda.as_cuda_array(E_dot),
            cuda.as_cuda_array(Hessian)
        )

        return Hessian, None


# ============================================================================
# Unified Interface - Automatically selects CPU or GPU implementation
# ============================================================================

class PathDTW:
    """
    Unified interface for Path DTW that automatically selects CPU or GPU implementation

    Usage:
        path_dtw = PathDTW(use_cuda=True)
        path_matrix = path_dtw(D_matrix, gamma)
    """

    def __init__(self, use_cuda=True):
        """
        Args:
            use_cuda: Whether to use GPU acceleration (default: True)
        """
        self.use_cuda = use_cuda

    def __call__(self, D, gamma):
        """
        Compute path DTW

        Args:
            D: Distance matrix [batch_size, N, N]
            gamma: Smoothing parameter

        Returns:
            Path matrix [N, N]
        """
        if self.use_cuda and D.device.type == 'cuda':
            # Check sequence length limit for CUDA
            N = D.shape[1]
            if N > 1024:
                print(f"Warning: Sequence length {N} > 1024, falling back to CPU implementation")
                return PathDTWBatch.apply(D, gamma)

            try:
                return PathDTWBatchCUDA.apply(D, gamma)
            except Exception as e:
                print(f"Warning: CUDA implementation failed ({e}), falling back to CPU")
                return PathDTWBatch.apply(D, gamma)
        else:
            return PathDTWBatch.apply(D, gamma)