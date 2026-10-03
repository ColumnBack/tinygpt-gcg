import os
import time

import numpy as np
from pathlib import Path


# ================================================================
# Tiny GPT
# NumPy only
#
# Architecture:
#
# token embedding
#       +
# positional embedding
#       |
#       v
# Transformer Block 1
#       |
#       v
# Transformer Block 2
#       |
#       ...
#       |
#       v
# final hidden representation
#       |
#       v
# tied LM head: logits = X @ E.T
#       |
#       v
# softmax
#       |
#       v
# cross entropy
#
# Transformer block:
#
# X
#  |
#  +--> Multi-Head Self-Attention --> O
#  |                                  |
#  +--------------- residual <--------+
#                    |
#                    v
#                   LN1
#                    |
#                    v
#                    Y
#                    |
#                    v
#                   FFN
#                    |
#                    v
#                    F
#                    |
#                    +---- residual with Y
#                    |
#                    v
#                   LN2
#                    |
#                    v
#                 X_next
# ================================================================


class TinyGPT:

    def __init__(
        self,
        vocab_size,
        max_context,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        eps=1e-5,
        seed=42
    ):

        assert d_model % n_heads == 0

        self.V = vocab_size
        self.max_context = max_context

        self.d = d_model
        self.H = n_heads
        self.dh = d_model // n_heads
        self.df = d_ff
        self.N = n_layers

        self.eps = eps

        rng = np.random.default_rng(seed)

        # ============================================================
        # Parameter dictionary
        # ============================================================

        self.p = {}

        # ------------------------------------------------------------
        # Token embedding
        #
        # E : V x d
        # ------------------------------------------------------------

        self.p["E"] = rng.normal(
            0.0,
            0.02,
            size=(self.V, self.d)
        )

        # ------------------------------------------------------------
        # Learned positional embedding
        #
        # P : max_context x d
        # ------------------------------------------------------------

        self.p["P"] = rng.normal(
            0.0,
            0.02,
            size=(self.max_context, self.d)
        )

        # ============================================================
        # Transformer blocks
        # ============================================================

        for n in range(self.N):

            # --------------------------------------------------------
            # Multi-head attention parameters
            #
            # WQ : d x dh
            # WK : d x dh
            # WV : d x dh
            # --------------------------------------------------------

            for h in range(self.H):

                self.p[f"WQ_{n}_{h}"] = rng.normal(
                    0.0,
                    1.0 / np.sqrt(self.d),
                    size=(self.d, self.dh)
                )

                self.p[f"WK_{n}_{h}"] = rng.normal(
                    0.0,
                    1.0 / np.sqrt(self.d),
                    size=(self.d, self.dh)
                )

                self.p[f"WV_{n}_{h}"] = rng.normal(
                    0.0,
                    1.0 / np.sqrt(self.d),
                    size=(self.d, self.dh)
                )

            # --------------------------------------------------------
            # Output projection
            #
            # WO : d x d
            # --------------------------------------------------------

            self.p[f"WO_{n}"] = rng.normal(
                0.0,
                1.0 / np.sqrt(self.d),
                size=(self.d, self.d)
            )

            # --------------------------------------------------------
            # First LayerNorm
            #
            # gamma1 : d
            # beta1  : d
            # --------------------------------------------------------

            self.p[f"gamma1_{n}"] = np.ones(self.d)
            self.p[f"beta1_{n}"] = np.zeros(self.d)

            # --------------------------------------------------------
            # Second LayerNorm
            #
            # gamma2 : d
            # beta2  : d
            # --------------------------------------------------------

            self.p[f"gamma2_{n}"] = np.ones(self.d)
            self.p[f"beta2_{n}"] = np.zeros(self.d)

            # --------------------------------------------------------
            # FFN
            #
            # U = Y W1 + b1
            # G = GELU(U)
            # F = G W2 + b2
            #
            # W1 : d x d_ff
            # W2 : d_ff x d
            # --------------------------------------------------------

            self.p[f"W1_{n}"] = rng.normal(
                0.0,
                1.0 / np.sqrt(self.d),
                size=(self.d, self.df)
            )

            self.p[f"b1_{n}"] = np.zeros(self.df)

            self.p[f"W2_{n}"] = rng.normal(
                0.0,
                1.0 / np.sqrt(self.df),
                size=(self.df, self.d)
            )

            self.p[f"b2_{n}"] = np.zeros(self.d)

        # ============================================================
        # Weight tying
        #
        # W_LM = E^T
        #
        # so W_LM is not stored as a separate parameter.
        # ============================================================


    # ================================================================
    # GELU
    #
    # The tanh approximation is used here to stay NumPy-only.
    #
    # GELU(x)
    # = 1/2 x [1 + tanh(c(x + 0.044715 x^3))]
    #
    # c = sqrt(2/pi)
    # ================================================================

    @staticmethod
    def gelu(x):

        c = np.sqrt(2.0 / np.pi)

        u = c * (
            x + 0.044715 * x**3
        )

        return 0.5 * x * (
            1.0 + np.tanh(u)
        )


    # ================================================================
    # GELU derivative
    #
    # GELU'(x)
    #
    # = 1/2(1+tanh(u))
    #   +
    #   1/2 x (1-tanh(u)^2)
    #       c(1+3*0.044715*x^2)
    # ================================================================

    @staticmethod
    def gelu_grad(x):

        c = np.sqrt(2.0 / np.pi)

        u = c * (
            x + 0.044715 * x**3
        )

        t = np.tanh(u)

        du_dx = c * (
            1.0 + 3.0 * 0.044715 * x**2
        )

        return (
            0.5 * (1.0 + t)
            +
            0.5 * x * (1.0 - t**2) * du_dx
        )


    # ================================================================
    # Row-wise softmax
    # ================================================================

    @staticmethod
    def softmax_rows(S):

        # numerical stability
        S_shift = (
            S
            -
            np.max(
                S,
                axis=1,
                keepdims=True
            )
        )

        exp_S = np.exp(S_shift)

        return (
            exp_S
            /
            np.sum(
                exp_S,
                axis=1,
                keepdims=True
            )
        )


    # ================================================================
    # LayerNorm forward
    #
    # R
    # |
    # v
    # mu = mean(R)
    #
    # C = R - mu
    #
    # var = mean(C^2)
    #
    # D = sqrt(var + eps)
    #
    # R_hat = C / D
    #
    # Y = R_hat * gamma + beta
    # ================================================================

    def layernorm_forward(
        self,
        R,
        gamma,
        beta
    ):

        # R : T x d

        mu = np.mean(
            R,
            axis=1,
            keepdims=True
        )

        # C : T x d
        C = R - mu

        # var : T x 1
        var = np.mean(
            C * C,
            axis=1,
            keepdims=True
        )

        # D : T x 1
        D = np.sqrt(
            var + self.eps
        )

        # R_hat : T x d
        R_hat = C / D

        # Y : T x d
        Y = R_hat * gamma + beta

        cache = (
            R,
            mu,
            C,
            var,
            D,
            R_hat,
            gamma
        )

        return Y, cache


    # ================================================================
    # LayerNorm backward
    #
    # Input:
    #
    # GY = dL/dY
    #
    # Output:
    #
    # GR
    # Ggamma
    # Gbeta
    #
    # Formula:
    #
    # GR_ij
    #
    # =
    # gamma_j / D_i
    #
    # * [
    #       GY_ij
    #       - mean_j(GY_i)
    #       - R_hat_ij mean_j(GY_i R_hat_i)
    #   ]
    # ================================================================

    def layernorm_backward(
        self,
        GY,
        cache
    ):

        (
            R,
            mu,
            C,
            var,
            D,
            R_hat,
            gamma
        ) = cache

 	# ------------------------------------------------------------
        # gamma gradient
        # G_Gamma = G_Y ⊙ R_hat
        #
        # G_gamma = 1_T^T G_Gamma
        # Ggamma_j
        # = sum_i GY_ij R_hat_ij
        # ------------------------------------------------------------

        Ggamma = np.sum(
            GY * R_hat,
            axis=0
        )

        # ------------------------------------------------------------
        # beta gradient
        # G_Beta = G_Y
	#
        # G_beta = 1_T^T G_Beta
        # Gbeta_j
        # = sum_i GY_ij
        # ------------------------------------------------------------

        Gbeta = np.sum(
            GY,
            axis=0
        )

        # ------------------------------------------------------------
        # G_Rhat
        #
        # Y = gamma ⊙ R_hat + beta
        #
        # G_Rhat = GY ⊙ gamma
        #
        # IMPORTANT:
        # gamma is feature-dependent.
        # Therefore gamma must be applied BEFORE taking
        # the mean over the feature dimension.
        # ------------------------------------------------------------

        G_Rhat = GY * gamma

        # ------------------------------------------------------------
        # mean(G_Rhat)
        #
        # mean over the feature dimension
        #
        # shape:
        # G_Rhat       : (T, d)
        # mean_GRhat   : (T, 1)
        # ------------------------------------------------------------

        mean_GRhat = np.mean(
            G_Rhat,
            axis=1,
            keepdims=True
        )

        # ------------------------------------------------------------
        # mean(G_Rhat * R_hat)
        #
        # shape:
        # G_Rhat * R_hat        : (T, d)
        # mean_GRhat_Rhat       : (T, 1)
        # ------------------------------------------------------------

        mean_GRhat_Rhat = np.mean(
            G_Rhat * R_hat,
            axis=1,
            keepdims=True
        )

        # ------------------------------------------------------------
        # GR
        #
        # GR =
        # 1 / D ⊙
        # [
        #     G_Rhat
        #     - mean(G_Rhat)
        #     - R_hat ⊙ mean(G_Rhat ⊙ R_hat)
        # ]
        #
        # D : (T, 1)
        # gamma : (d,)
        # ------------------------------------------------------------

        GR = (
            1.0 / D
        ) * (
            G_Rhat
            - mean_GRhat
            - R_hat * mean_GRhat_Rhat
        )

        return (
            GR,
            Ggamma,
            Gbeta
        )

    # ================================================================
    # One attention head forward
    #
    # Q = X WQ
    # K = X WK
    # V = X WV
    #
    # S = Q K^T / sqrt(dh)
    #
    # A = softmax(S + M)
    #
    # O = A V
    # ================================================================

    def one_head_forward(
        self,
        X,
        WQ,
        WK,
        WV
    ):

        T = X.shape[0]

        # ------------------------------------------------------------
        # Q, K, V
        # ------------------------------------------------------------

        Q = X @ WQ
        K = X @ WK
        V = X @ WV

        # ------------------------------------------------------------
        # S
        # ------------------------------------------------------------

        S = (
            Q @ K.T
        ) / np.sqrt(self.dh)

        # ------------------------------------------------------------
        # causal mask
        #
        # i >= j : allowed
        # j > i  : forbidden
        # ------------------------------------------------------------

        mask = np.triu(
            np.ones(
                (T, T),
                dtype=bool
            ),
            k=1
        )

        S_masked = np.where(
            mask,
            -1e9,
            S
        )

        # ------------------------------------------------------------
        # A
        # ------------------------------------------------------------

        A = self.softmax_rows(
            S_masked
        )

        # ------------------------------------------------------------
        # O
        # ------------------------------------------------------------

        O = A @ V

        cache = (
            X,
            Q,
            K,
            V,
            A,
            mask
        )

        return O, cache


    # ================================================================
    # One attention head backward
    #
    # O = A V
    #
    # A = softmax(S)
    #
    # S = QK^T / sqrt(dh)
    #
    # Q = XWQ
    # K = XWK
    # V = XWV
    #
    # Reverse:
    #
    # GO
    #  |
    #  +--> GA
    #  |
    #  +--> GV
    #
    # GA --> GS
    #
    # GS --> GQ, GK
    #
    # GQ, GK, GV --> GX
    # ================================================================

    def one_head_backward(
        self,
        GO,
        cache,
        WQ,
        WK,
        WV
    ):

        (
            X,
            Q,
            K,
            V,
            A,
            mask
        ) = cache

        scale = np.sqrt(
            self.dh
        )

        # ============================================================
        # O = A V
        #
        # dO = dA V + A dV
        #
        # GA = GO V^T
        # GV = A^T GO
        # ============================================================

        GA = (
            GO @ V.T
        )

        GV = (
            A.T @ GO
        )

        # ============================================================
        # Softmax backward
	# G_S=A⊙[G_A-((G_A⊙A)1T)1T.T]
        #
        # G_S,ik
        #
        # =
        # A_ik [
        #     G_A,ik
        #     -
        #     sum_j G_A,ij A_ij
        # ]
        # ============================================================

        row_dot = np.sum(
            GA * A,
            axis=1,
            keepdims=True
        )

        GS = (
            A
            *
            (
                GA - row_dot
            )
        )

        # Future positions do not participate.
        GS[mask] = 0.0

        # ============================================================
        # S = QK^T / sqrt(dh)
        #
        # dS
        # =
        # (dQ K^T + Q dK^T) / sqrt(dh)
        #
        # GQ = GS K / sqrt(dh)
        # GK = GS^T Q / sqrt(dh)
        # ============================================================

        GQ = (
            GS @ K
        ) / scale

        GK = (
            GS.T @ Q
        ) / scale

        # ============================================================
        # Q = XWQ
        # K = XWK
        # V = XWV
        #
        # GX:
        #
        # G_X^Q = GQ WQ^T
        # G_X^K = GK WK^T
        # G_X^V = GV WV^T
        # ============================================================

        GX_Q = (
            GQ @ WQ.T
        )

        GX_K = (
            GK @ WK.T
        )

        GX_V = (
            GV @ WV.T
        )

        GX = (
            GX_Q
            +
            GX_K
            +
            GX_V
        )

        # ============================================================
        # Weight gradients
        # ============================================================

        GWQ = (
            X.T @ GQ
        )

        GWK = (
            X.T @ GK
        )

        GWV = (
            X.T @ GV
        )

        return (
            GX,
            GWQ,
            GWK,
            GWV
        )


    # ================================================================
    # Transformer block forward
    #
    # X
    #
    # -> MHA
    #
    # -> R = X + O
    #
    # -> Y = LN(R)
    #
    # -> U = YW1 + b1
    #
    # -> G = GELU(U)
    #
    # -> F = GW2 + b2
    #
    # -> Z = Y + F
    #
    # -> X_next = LN(Z)
    # ================================================================

    def block_forward(
        self,
        X,
        n
    ):

        # ============================================================
        # Multi-head attention
        # ============================================================

        head_outputs = []
        head_caches = []

        for h in range(self.H):

            O_h, cache_h = self.one_head_forward(
                X,
                self.p[f"WQ_{n}_{h}"],
                self.p[f"WK_{n}_{h}"],
                self.p[f"WV_{n}_{h}"]
            )

            head_outputs.append(
                O_h
            )

            head_caches.append(
                cache_h
            )

        # ------------------------------------------------------------
        # Concatenate
        #
        # Ocat : T x d
        # ------------------------------------------------------------

        Ocat = np.concatenate(
            head_outputs,
            axis=1
        )

        # ------------------------------------------------------------
        # O = Ocat WO
        # ------------------------------------------------------------

        O = (
            Ocat
            @ self.p[f"WO_{n}"]
        )

        # ============================================================
        # First residual
        #
        # R = X + O
        # ============================================================

        R = X + O

        # ============================================================
        # First LayerNorm
        #
        # Y = LN(R)
        # ============================================================

        Y, ln1_cache = self.layernorm_forward(
            R,
            self.p[f"gamma1_{n}"],
            self.p[f"beta1_{n}"]
        )

        # ============================================================
        # FFN
        #
        # U = YW1 + b1
        # G = GELU(U)
        # F = GW2 + b2
        # ============================================================

        U = (
            Y
            @ self.p[f"W1_{n}"]
            +
            self.p[f"b1_{n}"]
        )

        G = self.gelu(U)

        F = (
            G
            @ self.p[f"W2_{n}"]
            +
            self.p[f"b2_{n}"]
        )

        # ============================================================
        # Second residual
        #
        # Z = Y + F
        # ============================================================

        Z = Y + F

        # ============================================================
        # Second LayerNorm
        #
        # X_next = LN(Z)
        # ============================================================

        X_next, ln2_cache = self.layernorm_forward(
            Z,
            self.p[f"gamma2_{n}"],
            self.p[f"beta2_{n}"]
        )

        cache = {
            "X": X,
            "head_caches": head_caches,
            "Ocat": Ocat,
            "ln1": ln1_cache,
            "Y": Y,
            "U": U,
            "G": G,
            "Z": Z,
            "ln2": ln2_cache
        }

        return (
            X_next,
            cache
        )


    # ================================================================
    # Transformer block backward
    #
    # Reverse order:
    #
    # X_next
    #   |
    #  LN2
    #   |
    #  Z = Y + F
    #   |
    #  FFN
    #   |
    #  Y
    #   |
    #  LN1
    #   |
    #  R = X + O
    #   |
    #  MHA
    #   |
    #  X
    # ================================================================

    def block_backward(
        self,
        GXnext,
        cache,
        n
    ):

        grads = {}

        # ============================================================
        # 1. Second LayerNorm
        #
        # X_next = LN(Z)
        #
        # GXnext -> GZ
        # ============================================================

        GZ, Ggamma2, Gbeta2 = (
            self.layernorm_backward(
                GXnext,
                cache["ln2"]
            )
        )

        grads[f"gamma2_{n}"] = Ggamma2
        grads[f"beta2_{n}"] = Gbeta2

        # ============================================================
        # 2. Second residual
        #
        # Z = Y + F
        #
        # Therefore:
        #
        # GY_direct = GZ
        # GF = GZ
        # ============================================================

        GY_direct = GZ
        GF = GZ

        # ============================================================
        # 3. FFN backward
        #
        # F = GW2 + b2
        # G = GELU(U)
        # U = YW1 + b1
        # ============================================================

        G = cache["G"]
        U = cache["U"]
        Y = cache["Y"]

        # ------------------------------------------------------------
        # F = GW2 + b2
        # ------------------------------------------------------------

        GW2 = (
            G.T @ GF
        )

        Gb2 = np.sum(
            GF,
            axis=0
        )

        GG = (
            GF
            @ self.p[f"W2_{n}"].T
        )

        # ------------------------------------------------------------
        # G = GELU(U)
        # ------------------------------------------------------------

        GU = (
            GG
            *
            self.gelu_grad(U)
        )

        # ------------------------------------------------------------
        # U = YW1 + b1
        # ------------------------------------------------------------

        GW1 = (
            Y.T @ GU
        )

        Gb1 = np.sum(
            GU,
            axis=0
        )

        GY_ffn = (
            GU
            @ self.p[f"W1_{n}"].T
        )

        grads[f"W2_{n}"] = GW2
        grads[f"b2_{n}"] = Gb2

        grads[f"W1_{n}"] = GW1
        grads[f"b1_{n}"] = Gb1

        # ------------------------------------------------------------
        # Y receives gradients from two paths:
        #
        # 1. residual path
        # 2. FFN path
        #
        # therefore gradients ADD.
        # ------------------------------------------------------------

        GY = (
            GY_direct
            +
            GY_ffn
        )

        # ============================================================
        # 4. First LayerNorm
        #
        # Y = LN(R)
        #
        # GY -> GR
        # ============================================================

        GR, Ggamma1, Gbeta1 = (
            self.layernorm_backward(
                GY,
                cache["ln1"]
            )
        )

        grads[f"gamma1_{n}"] = Ggamma1
        grads[f"beta1_{n}"] = Gbeta1

        # ============================================================
        # 5. First residual
        #
        # R = X + O
        #
        # Therefore:
        #
        # GX_direct = GR
        # GO = GR
        # ============================================================

        GX_direct = GR
        GO = GR

        # ============================================================
        # 6. Output projection
        #
        # O = Ocat WO
        #
        # G_Ocat = GO WO^T
        # G_WO   = Ocat^T GO
        # ============================================================

        Ocat = cache["Ocat"]

        GWO = (
            Ocat.T @ GO
        )

        GOcat = (
            GO
            @ self.p[f"WO_{n}"].T
        )

        grads[f"WO_{n}"] = GWO

        # ============================================================
        # 7. Split Ocat gradient into heads
        # ============================================================

        GX_attention = np.zeros_like(
            cache["X"]
        )

        for h in range(self.H):

            start = h * self.dh
            end = (h + 1) * self.dh

            GO_h = (
                GOcat[:, start:end]
            )

            # --------------------------------------------------------
            # Attention backward
            # --------------------------------------------------------

            GX_h, GWQ, GWK, GWV = (
                self.one_head_backward(
                    GO_h,
                    cache["head_caches"][h],
                    self.p[f"WQ_{n}_{h}"],
                    self.p[f"WK_{n}_{h}"],
                    self.p[f"WV_{n}_{h}"]
                )
            )

            GX_attention += GX_h

            grads[f"WQ_{n}_{h}"] = GWQ
            grads[f"WK_{n}_{h}"] = GWK
            grads[f"WV_{n}_{h}"] = GWV

        # ============================================================
        # 8. First residual gradient
        #
        # X receives:
        #
        # 1. direct residual path
        # 2. attention path
        #
        # ============================================================

        GX = (
            GX_direct
            +
            GX_attention
        )

        return (
            GX,
            grads
        )


    # ================================================================
    # Full forward
    #
    # token_ids
    #
    # -> E[token_ids]
    # -> + P
    # -> Transformer blocks
    # -> final X
    # -> logits
    #
    # Weight tying:
    #
    # W_LM = E^T
    #
    # therefore
    #
    # logits = X E^T
    # ================================================================

    def forward(
        self,
        token_ids
    ):

        token_ids = np.asarray(
            token_ids,
            dtype=np.int64
        )

        # ============================================================
        # Token embedding
        #
        # X_i = E[t_i,:]
        #
        # X : T x d
        # ============================================================

        X_token = (
            self.p["E"][token_ids]
        )

        return self.forward_embedded(
            X_token
        )


    # ================================================================
    # Forward from token embeddings
    #
    # X_token : T x d  (= E[token_ids])
    #
    # Split out of forward() so the input embeddings can be fed
    # directly. Used by the GCG gradient check, which perturbs
    # X_token to verify dL/dX_token by finite differences.
    # ================================================================

    def forward_embedded(
        self,
        X_token
    ):

        T = X_token.shape[0]

        assert (
            1 <= T <= self.max_context
        )

        # ============================================================
        # Learned positional embedding
        #
        # X^(0) = X_token + P
        # ============================================================

        X = (
            X_token
            +
            self.p["P"][:T]
        )

        # ============================================================
        # Transformer blocks
        # ============================================================

        block_caches = []

        for n in range(self.N):

            X, cache = (
                self.block_forward(
                    X,
                    n
                )
            )

            block_caches.append(
                cache
            )

        # ============================================================
        # LM head
        #
        # Weight tying:
        #
        # W_LM = E^T
        #
        # logits = X E^T
        #
        # X : T x d
        # E^T : d x V
        #
        # logits : T x V
        # ============================================================

        logits = (
            X
            @ self.p["E"].T
        )

        return (
            logits,
            X,
            block_caches
        )


    # ================================================================
    # Loss + full backward
    # ================================================================

    def loss_and_backward(
        self,
        input_ids,
        target_ids
    ):

        input_ids = np.asarray(
            input_ids,
            dtype=np.int64
        )

        target_ids = np.asarray(
            target_ids,
            dtype=np.int64
        )

        # ============================================================
        # Forward
        # ============================================================

        (
            logits,
            X_final,
            block_caches
        ) = self.forward(
            input_ids
        )

        T = len(input_ids)

        # ============================================================
        # Softmax
        # ============================================================

        logits_shift = (
            logits
            -
            np.max(
                logits,
                axis=1,
                keepdims=True
            )
        )

        exp_logits = np.exp(
            logits_shift
        )

        probs = (
            exp_logits
            /
            np.sum(
                exp_logits,
                axis=1,
                keepdims=True
            )
        )

        # ============================================================
        # Cross entropy
        #
        # L
        # =
        # -1/T sum_i log P[i, target_i]
        # ============================================================

        loss = -np.mean(
            np.log(
                probs[
                    np.arange(T),
                    target_ids
                ]
                + 1e-12
            )
        )

        # ============================================================
        # Reverse 1
        #
        # Cross Entropy + Softmax
        #
        # G_logits = (P - Y) / T
        # ============================================================

        Glogits = probs.copy()

	# P-Y, Y is one hot encoding
        Glogits[
            np.arange(T),
            target_ids
        ] -= 1.0

        Glogits /= T

        # ============================================================
        # Reverse 2
        #
	# L=X @ W_LM where E^T was used as Wlm (Weight Tying)
        # logits = X_final E^T
        #
        # Because:
        #
        # W_LM = E^T
        #
        # there are TWO routes into E:
        #
        # 1. LM head gradients
        # 2. token embedding gradients
        # ============================================================


        # ------------------------------------------------------------
        # 1. LM head gradients
        # dL/dX_final
        # 
	# G_X = G_L @ W_LM.T where W_LM is E^T 
        # G_X = G_logits E
        # ------------------------------------------------------------

        GX = (
            Glogits
            @ self.p["E"]
        )

        # ------------------------------------------------------------
        # dL/dE from LM head
        #
        # logits = X E^T
        #
        # G_E_LM = Glogits^T X
        # ------------------------------------------------------------

        GE_from_lm = (
            Glogits.T
            @ X_final
        )

        # Start gradient dictionary.
        grads = {}

        # ============================================================
        # Reverse 3
        #
        # Transformer blocks
        #
        # N -> N-1 -> ... -> 1
        # ============================================================

        for n in reversed(
            range(self.N)
        ):

            (
                GX,
                block_grads
            ) = self.block_backward(
                GX,
                block_caches[n],
                n
            )

            grads.update(
                block_grads
            )

        # ============================================================
        # Reverse 4
        #
        # X^(0) = E[token_ids] + P
        # ============================================================

        # ------------------------------------------------------------
        # 2. token embedding gradients
  	#
        # Multiple identical tokens:
        #
        # their gradients are summed.
        #
        # np.add.at performs exactly this accumulation.
        # ------------------------------------------------------------

        GE_from_embedding = (
            np.zeros_like(
                self.p["E"]
            )
        )

        np.add.at(
            GE_from_embedding,
            input_ids,
            GX
        )

        # ------------------------------------------------------------
        # Gradient of positional embedding
        # ------------------------------------------------------------

        GP = (
            np.zeros_like(
                self.p["P"]
            )
        )

        GP[:T] = GX

        # ------------------------------------------------------------
        # Weight tying:
        #
        # total GE
        #
        # =
        # embedding gradient
        # +
        # LM-head gradient
        # ------------------------------------------------------------

        GE = (
            GE_from_embedding
            +
            GE_from_lm
        )

        grads["E"] = GE
        grads["P"] = GP

        return (
            loss,
            grads
        )


    # ================================================================
    # Target loss + GCG token-swap scores (no one-hot)
    #
    # The input is plain token ids, exactly like forward() -- there is
    # no one-hot matrix. The classic GCG gradient w.r.t. the one-hot
    # input equals the embedding-input gradient projected onto the
    # vocabulary, so we compute it directly:
    #
    #   G[i, v] = (dL/dX_token[i]) . E[v]      ->    G = (dL/dX_token) E^T
    #
    # Only the target positions contribute to the loss:
    #
    #   L = -1/m sum_{i in positions} log P[i, target_i]
    #
    # Reverse:
    #
    #   G_logits (only target rows are non-zero)
    #   -> G_X_final = G_logits E
    #   -> blocks N..1
    #   -> G_X_token         (= dL/dX^(0), since X^(0) = X_token + P)
    #   -> G = G_X_token E^T                          (T x V)
    #
    # G[i, v] is the linearized change of L when the token at position i
    # is replaced by token v. GCG uses it to pick candidate swaps.
    #
    # (E also appears in the LM head via weight tying, but that route
    # reaches the parameter E, not the choice of input token, so it is
    # correctly excluded from the swap score.)
    # ================================================================

    def target_loss_and_token_grad(
        self,
        token_ids,
        positions,
        target_ids
    ):

        token_ids = np.asarray(
            token_ids,
            dtype=np.int64
        )

        positions = np.asarray(
            positions,
            dtype=np.int64
        )

        target_ids = np.asarray(
            target_ids,
            dtype=np.int64
        )

        m = len(positions)

        # ------------------------------------------------------------
        # Forward from token ids (same input path as tinyGPT)
        # ------------------------------------------------------------

        (
            logits,
            X_final,
            block_caches
        ) = self.forward(
            token_ids
        )

        # ------------------------------------------------------------
        # Softmax + cross entropy on the target rows only
        # ------------------------------------------------------------

        probs = self.softmax_rows(
            logits[positions]
        )

        loss = -np.mean(
            np.log(
                probs[
                    np.arange(m),
                    target_ids
                ]
                + 1e-12
            )
        )

        # ------------------------------------------------------------
        # G_logits = (P - Y) / m on the target rows, 0 elsewhere
        # ------------------------------------------------------------

        Glogits = np.zeros_like(
            logits
        )

        Glogits[positions] = probs

        Glogits[
            positions,
            target_ids
        ] -= 1.0

        Glogits /= m

        # ------------------------------------------------------------
        # LM head: G_X_final = G_logits E
        # ------------------------------------------------------------

        GX = (
            Glogits
            @ self.p["E"]
        )

        # ------------------------------------------------------------
        # Transformer blocks (parameter gradients are discarded)
        # ------------------------------------------------------------

        for n in reversed(
            range(self.N)
        ):

            GX, _ = self.block_backward(
                GX,
                block_caches[n],
                n
            )

        # ------------------------------------------------------------
        # GX is now dL/dX_token (X^(0) = X_token + P, so the
        # positional term drops out of the input gradient).
        #
        # Project onto the vocabulary embeddings:
        #
        #   G[i, v] = GX[i] . E[v]
        # ------------------------------------------------------------

        G = (
            GX
            @ self.p["E"].T
        )

        return (
            loss,
            G
        )


    # ================================================================
    # Text generation
    # ================================================================

    def generate(
        self,
        prompt_ids,
        id_to_token,
        max_new_tokens=10,
        temperature=0.7,
        stop_token="<EOS>"
    ):

        ids = list(
            prompt_ids
        )

        for _ in range(
            max_new_tokens
        ):

            # --------------------------------------------------------
            # Only the latest max_context tokens are visible.
            # --------------------------------------------------------

            context = (
                ids[-self.max_context:]
            )

            # --------------------------------------------------------
            # Forward
            # --------------------------------------------------------

            logits, _, _ = (
                self.forward(
                    context
                )
            )

            # --------------------------------------------------------
            # Last position
            # --------------------------------------------------------

            next_logits = (
                logits[-1]
                /
                max(
                    temperature,
                    1e-8
                )
            )

            # --------------------------------------------------------
            # Stable softmax
            # --------------------------------------------------------

            next_logits -= np.max(
                next_logits
            )

            p = np.exp(
                next_logits
            )

            p /= np.sum(p)

            # --------------------------------------------------------
            # Sampling
            # --------------------------------------------------------

            next_id = np.random.choice(
                len(p),
                p=p
            )

            ids.append(
                int(next_id)
            )

            # --------------------------------------------------------
            # Stop at EOS
            # --------------------------------------------------------

            if (
                id_to_token[next_id]
                ==
                stop_token
            ):
                break

        return ids


# ====================================================================
# Adam optimizer
# ====================================================================

class Adam:

    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        # First moment
        self.m = {
            k: np.zeros_like(v)
            for k, v in params.items()
        }

        # Second moment
        self.v = {
            k: np.zeros_like(v)
            for k, v in params.items()
        }


    def step(
        self,
        params,
        grads,
        clip_norm=1.0
    ):

        # ============================================================
        # Global gradient norm
        # ============================================================

        total_sq = 0.0

        for g in grads.values():

            total_sq += np.sum(
                g * g
            )

        total_norm = np.sqrt(
            total_sq
        )

        # ============================================================
        # Gradient clipping
        # ============================================================

        scale = min(
            1.0,
            clip_norm
            /
            (total_norm + 1e-12)
        )

        self.t += 1

        # ============================================================
        # Adam update
        # ============================================================

        for k in params:

            g = (
                grads[k]
                *
                scale
            )

            # First moment
            self.m[k] = (
                self.beta1 * self.m[k]
                +
                (1.0 - self.beta1) * g
            )

            # Second moment
            self.v[k] = (
                self.beta2 * self.v[k]
                +
                (1.0 - self.beta2) * (g * g)
            )

            # Bias correction
            m_hat = (
                self.m[k]
                /
                (1.0 - self.beta1 ** self.t)
            )

            v_hat = (
                self.v[k]
                /
                (1.0 - self.beta2 ** self.t)
            )

            # Parameter update
            params[k] -= (
                self.lr
                *
                m_hat
                /
                (
                    np.sqrt(v_hat)
                    +
                    self.eps
                )
            )


# ====================================================================
# Dataset
# ====================================================================

def build_dataset(
    sentences
):

    # ---------------------------------------------------------------
    # Special tokens
    # ---------------------------------------------------------------

    special_tokens = [
        "<BOS>",
        "<EOS>",
        "<UNK>"
    ]

    # ---------------------------------------------------------------
    # Collect words
    # ---------------------------------------------------------------

    words = []

    for sentence in sentences:

        # Case is preserved (no .lower()) so that structured targets such as
        # {"action":"REFUND", ...} keep their exact spelling. All prompt words
        # in the corpus are already lowercase, so the vocabulary does not split.
        words.extend(
            sentence.split()
        )

    # ---------------------------------------------------------------
    # Vocabulary
    # ---------------------------------------------------------------

    vocabulary = (
        special_tokens
        +
        sorted(
            set(words)
        )
    )

    token_to_id = {
        token: i
        for i, token in enumerate(
            vocabulary
        )
    }

    id_to_token = {
        i: token
        for token, i in token_to_id.items()
    }

    # ---------------------------------------------------------------
    # Convert each sentence to IDs
    #
    # sentence:
    #
    # "i like cats"
    #
    # becomes:
    #
    # <BOS> i like cats <EOS>
    # ---------------------------------------------------------------

    data = []

    for sentence in sentences:

        tokens = (
            ["<BOS>"]
            +
            sentence.split()
            +
            ["<EOS>"]
        )

        ids = [
            token_to_id.get(
                token,
                token_to_id["<UNK>"]
            )
            for token in tokens
        ]

        data.append(
            ids
        )

    return (
        token_to_id,
        id_to_token,
        data
    )


# ====================================================================
# Decode token IDs
# ====================================================================

def decode(
    ids,
    id_to_token
):

    words = []

    for i in ids:

        token = id_to_token[
            int(i)
        ]

        if token == "<BOS>":
            continue

        if token == "<EOS>":
            break

        words.append(
            token
        )

    return " ".join(
        words
    )


# ====================================================================
# Training state (save / load)
#
# Parameters + Adam state + number of completed epochs are saved
# in one file, so the weights survive an interrupted run
# and training can resume on the next run.
#
# File keys:
#
#   p.<name>    model parameter
#   m.<name>    Adam first moment
#   v.<name>    Adam second moment
#   adam_t      Adam step count
#   epoch       last fully completed epoch
#   sentences   training sentences (retrain if they change)
# ====================================================================

def save_state(
    path,
    model,
    optimizer,
    epoch,
    sentences
):

    path = Path(path)

    arrays = {}

    for k in model.p:

        arrays[f"p.{k}"] = model.p[k]
        arrays[f"m.{k}"] = optimizer.m[k]
        arrays[f"v.{k}"] = optimizer.v[k]

    # ---------------------------------------------------------------
    # Write to a temp file first and swap it in, so the existing
    # file is never corrupted if the process dies mid-save.
    # ---------------------------------------------------------------

    tmp_path = path.with_suffix(".tmp")

    with open(tmp_path, "wb") as f:

        np.savez(
            f,
            adam_t=optimizer.t,
            epoch=epoch,
            sentences=np.array(sentences),
            **arrays
        )

    # ---------------------------------------------------------------
    # On Windows, antivirus / search indexing can briefly lock a newly
    # written file, making the swap fail with PermissionError. Retry.
    # ---------------------------------------------------------------

    for attempt in range(20):

        try:

            os.replace(
                tmp_path,
                path
            )

            return

        except PermissionError:

            time.sleep(0.1 * (attempt + 1))

    raise PermissionError(
        f"Could not save {path}. "
        f"(the latest weights are in {tmp_path})"
    )


def load_state(
    path,
    model,
    optimizer,
    sentences
):

    # ---------------------------------------------------------------
    # Load the saved state into model / optimizer and
    # return the number of completed epochs.
    #
    # None if the file is missing or the sentences / architecture changed.
    # ---------------------------------------------------------------

    path = Path(path)

    if not path.exists():
        return None

    with np.load(path) as f:

        if (
            "sentences" not in f.files
            or
            list(f["sentences"]) != list(sentences)
        ):
            return None

        for k in model.p:

            if (
                f"p.{k}" not in f.files
                or
                f[f"p.{k}"].shape != model.p[k].shape
            ):
                return None

        for k in model.p:

            model.p[k] = f[f"p.{k}"].copy()
            optimizer.m[k] = f[f"m.{k}"].copy()
            optimizer.v[k] = f[f"v.{k}"].copy()

        optimizer.t = int(f["adam_t"])

        return int(f["epoch"])



# ====================================================================
# Load model for inference
#
# Rebuild the model from model.npz alone.
#
#   vocabulary   : saved training sentences -> build_dataset (same as in training)
#   architecture : inferred from the parameter shapes
#
#     vocab_size  = E.shape[0]
#     d_model     = E.shape[1]
#     max_context = P.shape[0]
#     n_heads     = number of WQ_0_* entries
#     d_ff        = W1_0.shape[1]
#     n_layers    = number of WO_* entries
# ====================================================================

def load_model(
    path
):

    with np.load(path) as f:

        p = {
            k[len("p."):]: f[k].copy()
            for k in f.files
            if k.startswith("p.")
        }

        sentences = [
            str(s)
            for s in f["sentences"]
        ]

        epoch = int(f["epoch"])

    (
        token_to_id,
        id_to_token,
        _
    ) = build_dataset(
        sentences
    )

    model = TinyGPT(

        vocab_size=p["E"].shape[0],

        max_context=p["P"].shape[0],

        d_model=p["E"].shape[1],

        n_heads=sum(
            k.startswith("WQ_0_")
            for k in p
        ),

        d_ff=p["W1_0"].shape[1],

        n_layers=sum(
            k.startswith("WO_")
            for k in p
        )
    )

    assert model.V == len(token_to_id)

    for k in model.p:

        assert model.p[k].shape == p[k].shape, k

        model.p[k] = p[k]

    return (
        model,
        token_to_id,
        id_to_token,
        sentences,
        epoch
    )
