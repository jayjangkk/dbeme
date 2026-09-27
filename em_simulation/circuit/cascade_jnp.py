"""A DBEME cascade as a ``jnp`` function of its section lengths - `tasks/15` §5.4 Route B.

The scattering route of `SingleEME._calc_Smatrix_direct` is
``P_0 * I_0 * P_1 * I_1 * ...`` (Redheffer star products): propagation
through section ``k``, then the interface at its end.  The interface matrices
``I_k`` come from the stored overlaps and carry the production projection and
column cap; the propagation blocks ``P_k = diag(exp(i beta dz_k))`` (both
directions, scattering form: nothing grows) are the only place the section
lengths enter.  So a device's S-matrix is exactly differentiable in every
``dz_k`` with no solver derivative - the axes that cost zero mode solves are
the differentiable ones (`studies/sirac/differentiable.py`, report 07 §18).

That module is autograd on numpy; this one is the same algebra in ``jnp`` so
a path device's length and longitudinal shape can be Adam variables *in the
same graph* as a sax circuit (a RAC's section-II length inside the
interleaver's loss).  :meth:`CascadeJnp.from_path` takes the matrices
straight from the production propagator, so the two cannot drift apart;
`tests/test_cascade_jnp.py` asserts the S-matrix equals `lumped_smatrix`
and the gradient equals finite differences.
"""

import numpy as np

import jax
import jax.numpy as jnp

__all__ = ["redheffer", "propagation_block", "CascadeJnp"]


def redheffer(a, b):
    """Redheffer star product of two ``(2n, 2n)`` scattering matrices (``a`` upstream)."""
    n = a.shape[-1] // 2
    a11, a12, a21, a22 = a[:n, :n], a[:n, n:], a[n:, :n], a[n:, n:]
    b11, b12, b21, b22 = b[:n, :n], b[:n, n:], b[n:, :n], b[n:, n:]
    eye = jnp.eye(n, dtype=a.dtype)
    c1 = b11 @ jnp.linalg.inv(eye - a12 @ b21)
    c2 = a22 @ jnp.linalg.inv(eye - b21 @ a12)
    top = jnp.concatenate([c1 @ a11, c1 @ a12 @ b22 + b12], axis=1)
    bottom = jnp.concatenate([a21 + c2 @ b21 @ a11, c2 @ b22], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def propagation_block(beta, dz):
    """``diag(exp(i beta dz))`` in both blocks (scattering form)."""
    phase = jnp.exp(1j * beta * dz)
    n = phase.shape[0]
    block = jnp.diag(phase)
    zero = jnp.zeros((n, n), dtype=block.dtype)
    return jnp.concatenate([jnp.concatenate([block, zero], axis=1), jnp.concatenate([zero, block], axis=1)], axis=0)


class CascadeJnp:
    """Frozen interface matrices; the section lengths are the differentiable input.

    :param interfaces: ``(n_steps, 2N, 2N)`` interface scattering matrices of
        the production propagator (``SingleEME._calc_interface_Smatrix()``).
    :param beta_forward: ``(>= n_steps, N)`` propagation constants of the
        forward modes, section by section (``propagator.beta_forward``).
    """

    def __init__(self, interfaces, beta_forward):
        self.interfaces = jnp.asarray(np.asarray(interfaces, dtype=complex))
        beta = np.asarray(beta_forward, dtype=complex)
        self.n_steps = int(self.interfaces.shape[0])
        self.n_modes = int(self.interfaces.shape[1] // 2)
        self.beta = jnp.asarray(beta[: self.n_steps, : self.n_modes])
        self._smatrix = jax.jit(self._smatrix_impl)

    @classmethod
    def from_path(cls, path, method="direct"):
        """Build from a solved path; returns ``(cascade, delta_zs)``."""
        from em_simulation import EME

        eme = EME(path, force_unitary=False)
        prop = eme.propagator
        prop.calc_Smatrix(method=method)
        dz = np.asarray(path.output_data["EME_delta_zs"], dtype=float)
        return cls(prop._calc_interface_Smatrix(), np.asarray(prop.beta_forward)), dz

    def _smatrix_impl(self, delta_zs):
        delta_zs = jnp.asarray(delta_zs, dtype=jnp.float64)[: self.n_steps]

        def body(total, inputs):
            interface, beta, dz = inputs
            total = redheffer(total, propagation_block(beta, dz))
            total = redheffer(total, interface)
            return total, None

        start = propagation_block(self.beta[0], delta_zs[0])
        start = redheffer(start, self.interfaces[0])
        total, _ = jax.lax.scan(body, start, (self.interfaces[1:], self.beta[1:], delta_zs[1:]))
        return total

    def smatrix(self, delta_zs):
        """``(2N, 2N)`` device S-matrix, ``[b_out; b_in] = S [a_in; a_out]``, for the given section lengths."""
        return self._smatrix(jnp.asarray(delta_zs, dtype=jnp.float64))

    def transmission(self, delta_zs):
        """The forward ``(N, N)`` block (out <- in) in tracked-branch order."""
        return self.smatrix(delta_zs)[: self.n_modes, : self.n_modes]
