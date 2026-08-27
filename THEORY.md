Consider a vector $\phi_n^m$ of coefficients of $R_n^m$ with

\begin{align}
(L_\phi)_n^m R_n^m(\vec{r}) &= \phi_n^m \tilde{R}_n^m Y_n^m(\vec{r}) r^n
\end{align}
Reconstructing the normalization like we did before, we have:

\begin{align}
i^{|m|}(-1)^m \sqrt{\frac{(2n+1)(n-|m|)!}{4\pi(n+|m|)!}} \tilde{R}_n^m &= (-1)^n \frac{i^{|m|}}{(n+|m|)!}\\
\tilde{R}_n^m &= \sqrt{\frac{4\pi}{(2n+1)(n+|m|)!(n-|m|)!}}
\end{align}
which implies the rotation-invariant norm:

\begin{align}\label{eq:local-norm}
\mathcal{L}_n &= \sqrt{\sum \limits_{m=-n}^n \left[ \phi_n^m \tilde{R}_n^m \right]^2}
\end{align}
Then, rotating into a $z$-aligned coordinate system with the point of maximum error such that $|m|>0$ terms vanish, we arrive at the following for a local expansion:

\begin{align}
\varepsilon_\phi &\lesssim \frac{\mathcal{L}_{p}}{\tilde{R}_{p}^0} \frac{r^{p}}{p!}
\end{align}
We can conservatively choose $r$ as the distance from the local expansion center to the farthest corner of the cell.

It is often more helpful to control the error in the induced vector field $\vec{v}$ from Eq.~\ref{eq:scalar-plus-vector} rather than the potential itself. Upper bound estimates can be derived as
\begin{align}
\left|\vec{\varepsilon}_{R}\right| &\lesssim \sqrt{3} \frac{\mathcal{L}_{p}^{(\tilde{\phi})}}{\tilde{R}_{p}^0} \frac{r^{p-1}}{(p-1)!} + \sqrt{3} \frac{\mathcal{L}_{p}^{(\chi)}}{\tilde{R}_{p}^0} \frac{r^{p}}{p!}\label{eq:total-local-error}\\
\left|\vec{\varepsilon}_{S}\right| &\lesssim \sqrt{3} \frac{\mathcal{M}_{p-1}}{\tilde{S}_{p-1}^0} \frac{p!}{r^{p+1}} + \sqrt{3} \frac{\mathcal{M}_{p}^{(\chi)}}{\tilde{S}_{p}^0} \frac{p!}{r^{p+1}}\label{eq:total-multipole-error}
\end{align}
where $\left|\vec{\varepsilon}_{R}\right|$ is the magnitude of the vector error experienced by a local expansion, and $\left|\vec{\varepsilon}_{S}\right|$ is for a multipole expansion.
