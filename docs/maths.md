# pyuvimage: the mathematics in one place

Notation follows the central relationship

$$
V'(u,v) = V(u,v)\cdot S(u,v)
\quad\overset{\textsc{FT}}{\Longleftrightarrow}\quad
I'(x,y) = I(x,y) \ast B(x,y),
$$

with the w-tilde term $\tilde W$ named as in PyAutoArray.

## 0. Symbols

| symbol | meaning |
|---|---|
| $k = 1\dots N_{\rm vis}$ | visibility index; $(u_k, v_k)$ in wavelengths, per channel at that channel's frequency |
| $V'_k$ | measured (sampled) visibility, complex |
| $V(u,v) = \int I(x,y)\,e^{-2\pi i(ux+vy)}\,dx\,dy$ | sky visibilities |
| $\sigma_k$, $w_k = 1/\sigma_k^2$ | per-visibility noise (real = imaginary, pooled) and natural weight |
| $\Sigma w = \sum_k w_k$ | total weight; $\sigma_{\rm map} = 1/\sqrt{\Sigma w}$ is the dirty-image rms |
| $S(u,v) = \sum_k w_k\,\delta(u-u_k)\,\delta(v-v_k)$ | weighted sampling function |
| $B = \textsc{FT}^{-1}[S]/\Sigma w$ | dirty beam, peak 1 |
| $I' = \textsc{FT}^{-1}[V' S]/\Sigma w$ | dirty image of the data, Jy/beam |
| $p, q = 1\dots N_{\rm img}$ | pixels of the image (product) grid, positions $\mathbf x_p$ |
| $i, j = 1\dots N_{\rm mesh}$ | pixels of the model mesh ($N_{\rm img} = 4N_{\rm mesh}$ at the default oversampling of 2) |
| $s$ | the mesh solution (vector, $N_{\rm mesh}$) |
| $M$ | mapping matrix, mesh → image grid ($N_{\rm img}\times N_{\rm mesh}$, sparse); $I = Ms$ |
| $T_{kp} = e^{-2\pi i\,\mathbf u_k\cdot\mathbf x_p}$ | Fourier transform of image pixels to the visibility positions |
| $A = TM$ | the response matrix ($N_{\rm vis}\times N_{\rm mesh}$): column $i$ is the visibilities that mesh pixel $i$ alone would produce at unit brightness, so $As$ is the model's visibilities |
| $W = {\rm diag}(w_k)$ | the weights as a matrix: $w_k$ on the diagonal, zero elsewhere |
| $H = \lambda C^{-1}$ | the prior's penalty matrix: strength $\lambda$; $C$ (Matérn etc.) says how alike nearby pixels are expected to be |
| $P(x,y)$ | primary beam, Gaussian, FWHM $1.13\,\lambda_{\rm obs}/D$, centred on the pointing |


### How to read the maths

- **Vectors and matrices are lists and tables of numbers.** $s$ lists the brightness of every mesh pixel; $F$ is a table with one row and one column per mesh pixel; $Ms$ means "apply the table $M$ to the list $s$".
- **The model is linear:** doubling the sky doubles every model visibility. That is why the best image comes from solving one set of linear equations rather than from a slow trial-and-error search.
- **$^T$ and $^\dagger$** swap a table's rows and columns ($^\dagger$ also takes complex conjugates). $s^TFs$ is a single number: the sum over all pixel pairs of $s_iF_{ij}s_j$.
- **The inverse $(F+H)^{-1}$** is the matrix version of dividing. It is not needed to find the model, only its uncertainties.
- **$\ln\det$** (log-determinant) measures how much a matrix shrinks or stretches the space of possible models.

---

## 1. Sky-model inference with an explicit transform (dense path)

**Forward model.** Two steps take the mesh values $s$ to model visibilities. $M$ spreads the mesh onto the image grid, $I = Ms$; then $T$ Fourier-transforms the image to each observed uv point:

$$
V^{\rm m}_k = \sum_p T_{kp}\,I_p = (TMs)_k = (As)_k .
$$

$A = TM$ is the **response matrix**, one row per visibility and one column per mesh pixel. Column $i$ is the set of visibilities that mesh pixel $i$ alone would produce at unit brightness, and $As$ adds up every pixel's contribution.

The product $T I$ is evaluated by a direct DFT (exact) or a NUFFT. A NUFFT approximates $T$ to a set tolerance by gridding onto an oversampled grid followed by an FFT. Either way, the model is **linear in $s$**.

**Data model.** Each measured visibility is the model visibility plus noise:

$$
V'_k = (As)_k + n_k,\qquad {\rm Re}\,n_k,\ {\rm Im}\,n_k \sim \mathcal N(0,\sigma_k^2)\ \text{independently}.
$$

The noise $\sigma_k$ is estimated from the data (§5); everything below rests on this statement.

**Likelihood.** The probability of the data given a model follows directly from the data model:

$$
\mathcal L(s) = P(V'\,|\,s) \propto e^{-\chi^2(s)/2},\qquad
\chi^2(s) = \sum_k w_k\,\big|V'_k - (As)_k\big|^2 = (V'-As)^{\dagger}\,W\,(V'-As).
$$

**Prior.** A zero-mean Gaussian process on the mesh:

$$
P(s\,|\,\lambda) \propto e^{-\frac12 s^T H s},\qquad H = \lambda\,C^{-1}.
$$

**Posterior.** The best model balances the fit to the data against the prior: it maximises $-\tfrac12\big[\chi^2(s) + s^THs\big]$. Because the model is linear in $s$, this is a quadratic in $s$ (like a parabola, with one lowest point), and its lowest point is the solution of one set of linear equations

$$
\boxed{(F + H)\,\hat s = D}
\qquad F = {\rm Re}\big(A^{\dagger} W A\big),\quad D = {\rm Re}\big(A^{\dagger} W V'\big).
$$

- $F$ records how strongly the data pin down each mesh pixel, and how much each pair of pixels is constrained together. (Statisticians call it the Fisher matrix; it is the curvature of $\chi^2$.)
- $D$ is the data vector: the weighted dirty image as seen by each mesh pixel.
- Read the equation as "data constraint plus prior, applied to the model, must reproduce the dirty image".
- $(F+H)^{-1}$, the matrix inverse, gives the uncertainty of $\hat s$: each pixel's variance on its diagonal, and how pixel errors are correlated off it.

**Positivity.** By default $\hat s = \arg\min_{s\ge 0}\big[\tfrac12 s^T(F+H)s - s^TD\big]$, solved by non-negative least squares (NNLS). This is the most probable model once negative skies are ruled out. The solver decides which pixels are "on" (positive) and which are held at zero, switching pixels until no switch improves the fit; for the "on" pixels it solves the same equation as above.

**Cost.** Forming $A$ costs $N_{\rm vis}\times N_{\rm mesh}$ complex numbers of memory (about 44 bytes each, as measured); forming $F$ takes time in proportion to $N_{\rm vis}\times N_{\rm mesh}^2$.

---

## 2. The likelihood and the expanded $\chi^2$

Multiply out the brackets. The two middle terms are the same number (each is the data weighted against the model), so they combine:

$$
\chi^2(s) = V'^{\dagger}WV' - 2\,{\rm Re}\big(s^TA^{\dagger}WV'\big) + s^T\,{\rm Re}(A^{\dagger}WA)\,s
$$

$$
\boxed{\chi^2(s) = s^TFs - 2\,s^TD + c},\qquad c = \sum_k w_k\,|V'_k|^2 .
$$

The full log-likelihood adds the noise normalisation, which counts two real numbers per visibility:

$$
\ln\mathcal L(s) = -\tfrac12\chi^2(s) - \sum_k \ln\!\big(2\pi\sigma_k^2\big).
$$

**Why the expansion is faster.** $F$, $D$ and $c$ do not depend on $s$. They are computed once, and then:

| | direct $\chi^2$ | expanded $\chi^2$ |
|---|---|---|
| per evaluation | predict all $N_{\rm vis}$ visibilities, sum all residuals: work $\propto N_{\rm vis}N_{\rm mesh}$ | $s^TFs$, $s^TD$: work $\propto N_{\rm mesh}^2$, **independent of $N_{\rm vis}$** |
| direction to improve the model (gradient) | another transform | $\nabla\chi^2 = 2(Fs - D)$ |
| data needed | all visibilities, every call | none after $F, D, c$ are built |

The hyperparameter search, the NNLS iterations and the systematic-window solves all evaluate this same expression, often dozens of times per fit. Each is now a mesh-sized operation. Because the visibilities are not needed afterwards, the data can also be streamed: read in chunks, accumulate, discard.

---

## 3. The w-tilde term ($\tilde W$, the sparse operator)

Building $F = {\rm Re}(A^\dagger W A)$ directly still takes time in proportion to $N_{\rm vis}\times N_{\rm mesh}^2$. Split $A = TM$:

$$
F = M^T\,\tilde W\,M,\qquad
\tilde W_{pq} = {\rm Re}\big(T^\dagger W T\big)_{pq} = \sum_k w_k\cos\!\big(2\pi\,\mathbf u_k\cdot(\mathbf x_p-\mathbf x_q)\big) = \Sigma w\;B(\mathbf x_p-\mathbf x_q).
$$

**$\tilde W$ is the dirty beam.** It is unnormalised, peaking at $\Sigma w$, and evaluated at the separation of two pixels. It therefore depends only on the separation $\mathbf x_p - \mathbf x_q$, not on where the two pixels are: one image of the dirty beam over all possible separations — a $(2N_y-1)\times(2N_x-1)$ grid, twice the image size because separations run from $-N$ to $+N$ — specifies all of it, and applying $\tilde W$ to an image is a **convolution with the dirty beam**:

$$
(\tilde W I)_p = \Sigma w\,(I\ast B)_p .
$$

The data side likewise reduces to the dirty image:

$$
D = M^T\,{\rm Re}\big(T^\dagger W V'\big) = \Sigma w\;M^T I' .
$$

**$\chi^2$ in the image plane.** Substituting into the expanded form, with $I = Ms$:

$$
\boxed{\chi^2(s) = I^T\tilde W I - 2\,\Sigma w\;I^T I' + c
= \Sigma w\Big[\textstyle\sum_p I_p\,(I\ast B)_p - 2\sum_p I_p\,I'_p\Big] + c.}
$$

This is the central relationship used as a fit. The model's dirty image $I\ast B$ is compared with the data's dirty image $I'$. The visibilities enter only through $B$ (via $\tilde W$), $I'$ and $c$.

**Building it (one pass over the data, streamable).** Everything is accumulated chunk by chunk:

- $\tilde W$ on the doubled grid: the weights gridded and Fourier-transformed to an image with one non-uniform FFT (NUFFT, the standard fast transform for irregularly sampled visibilities), $\tilde W(\boldsymbol\Delta) = {\rm Re}\sum_k w_k\,e^{+2\pi i\,\mathbf u_k\cdot\boldsymbol\Delta}$.
- $\Sigma w\,I'$: the same transform applied to $w_kV'_k$.
- $c = \sum_k w_k|V'_k|^2$ and $\Sigma w$: running sums.

$N_{\rm vis}$ appears only in this pass. $F$ is then assembled column by column as $F_{:,i} = M^T\big(\tilde W M_{:,i}\big)$, each product being an FFT convolution on the doubled grid.

**Inference.** It is unchanged: $(F+H)\hat s = D$, with or without positivity. Only the construction of $F$ and $D$ differs.

| | dense (§1) | w-tilde (§3) |
|---|---|---|
| memory | $\propto N_{\rm vis}N_{\rm mesh}$ | $\propto N_{\rm img} + N_{\rm mesh}^2$, flat in $N_{\rm vis}$ |
| one-off time | $\propto N_{\rm vis}N_{\rm mesh}^2$ | $\propto N_{\rm vis}$ for the pass, plus one FFT per mesh column to build $F$ |
| per trial | one solve, work $\propto N_{\rm mesh}^3$ | the same |

**Condition: one weight per visibility.** With separate real and imaginary weights, the coefficient of $I_pI_q$ is

$$
\tfrac12(w_{\rm re}+w_{\rm im})\cos(\theta_p-\theta_q) + \tfrac12(w_{\rm re}-w_{\rm im})\cos(\theta_p+\theta_q),
\qquad \theta_p = 2\pi\,\mathbf u_k\cdot\mathbf x_p .
$$

The second term depends on $\mathbf x_p+\mathbf x_q$, so it is not a convolution. The sparse path therefore pools $\sigma_{\rm re}$ and $\sigma_{\rm im}$ into one $\sigma_k$. This is physically expected: both components come from the same integration.

---

## 4. Including the primary beam

What the array measures is the visibilities of the **apparent** sky, $I_{\rm app} = P\,I_{\rm true}$:

$$
V'(u,v) = \textsc{FT}\big[P\,I_{\rm true}\big]\cdot S(u,v),\qquad I' = (P\,I_{\rm true})\ast B .
$$

So in §1–3, $I$ is $I_{\rm app}$.

**Default: the primary-beam correction afterwards.** Solve for the apparent sky with the prior on the apparent sky, then divide:

$$
(F+H)\,\hat s_{\rm app} = D,\qquad I_{\rm true} = M\hat s_{\rm app}/P\quad(\text{blanked where } P<0.1).
$$

**Primary beam in the model (`--pb-in-model`).** Solve for the true sky $s_t$. $P$ varies on the primary-beam scale (~20″), so it is constant across one mesh pixel and can be written as $P_m = {\rm diag}\big(P(\mathbf x_i)\big)$ on the mesh:

$$
I_{\rm app} = M P_m\,s_t,\qquad A' = A P_m,
$$

$$
F' = P_m F P_m,\qquad D' = P_m D,\qquad \chi^2(s_t) = s_t^TF's_t - 2s_t^TD' + c .
$$

$\tilde W$, $I'$ and $c$ are unchanged, because the primary beam does not enter the dirty beam. With the prior on the true sky:

$$
\boxed{(P_mFP_m + H)\,\hat s_t = P_mD}
\quad\Longleftrightarrow\quad
\big(F + P_m^{-1}HP_m^{-1}\big)\,\hat s_{\rm app} = D,\quad \hat s_{\rm app} = P_m\hat s_t .
$$

So folding $P$ into $F$ is exactly a change of prior: on the apparent sky the prior's penalty grows as $1/P^2$ towards the edge. The two treatments differ only where $P<1$:

- **Default:** a prior on the apparent sky is, on the true sky, a prior whose variance grows as $1/P^2$. It is permissive at the edge: unbiased but noisy, and the division by $P$ amplifies that noise.
- **In the model:** a stationary prior on the true sky. It is stricter at the edge: less noise, but faint edge emission is pulled lower.

The evidence compares the two directly (§5).

$P_m$ is floored at 0.1. An adaptive prior's brightness map comes from a first pass on the apparent sky, so it is divided by $P_m$ as well. For **mosaics** (not implemented) each pointing $j$ has its own primary beam and its own $\tilde W_j$, and the sum $F = \sum_j M^TP_j\tilde W_jP_jM$ is where the primary beam must be inside $F$.

---

## 5. Other parts that complete the method

**Evidence.** For hyperparameters $\theta$ (coefficient, correlation length, ...):

$$
\ln Z(\theta) = -\tfrac12\Big[\chi^2(\hat s) + \hat s^TH\hat s + \ln\det(F+H) - \ln\det H\Big] - \sum_k\ln(2\pi\sigma_k^2).
$$

This is the probability of the data under a given prior, averaged over every model the prior allows. It rewards fitting the data ($\chi^2$) and penalises a prior that needs an implausible model ($\hat s^TH\hat s$) or is flexible enough to fit anything (the two $\ln\det$ terms, which measure how much the data narrow down the models the prior allows). Because it averages over all models, it does not depend on whether the model is written as the apparent or the true sky (§4).

**Choosing $\lambda$.** One of three criteria. `auto` takes structure when $N_d/N_{\rm mesh} \ge 10$ (with $N_d = 2N_{\rm vis}$ data points) and discrepancy otherwise.

- *Discrepancy:* $\chi^2(\hat s) = N_d$, using the target raised to the measured $\chi^2$ floor when positivity makes $N_d$ unreachable.
- *Structure ratio:* make the residual dirty image white,

$$
\rho = \frac{{\rm rms}\big[(I' - I\ast B)/\sigma_{\rm map}\big]}{\sqrt{\chi^2/N_d}} = 1,
$$

  where $I'-I\ast B = (\Sigma w\,I' - \tilde W I)/\Sigma w$ is the residual dirty image. Incoherent residuals give $\rho\approx1$; missed sky adds coherently and gives $\rho>1$; a model that has absorbed noise gives $\rho<1$.
- *Evidence:* maximise $\ln Z$.

**Uncertainty.** All derived from $\mathcal C = (F+H)^{-1}$, the uncertainty matrix of the mesh solution (each pixel's variance on its diagonal, correlations between pixel errors elsewhere):

- *Statistical, per pixel:* $\sqrt{{\rm diag}(M\,\mathcal C\,M^T)}$ — $\mathcal C$ carried onto the image grid; ${\rm diag}$ keeps each pixel's own variance.
- *At the restoring beam $G$ (`snr.fits`):* $\sqrt{{\rm diag}(GM\,\mathcal C\,M^TG^T)}$, so that S/N is $(G\ast I)$ over this.
- *Prior systematic:* the spread of $M\hat s(\lambda')$ over the admissible window of $\lambda'$. The window is set where $\chi^2$, or for structure fits $\rho$, stays within its own noise scatter of the fitted value. This is added in quadrature.
- *Apertures:* use the full $g^TM\mathcal CM^Tg$, where $g$ marks the aperture's pixels. This includes the correlations between pixel errors; a quadrature sum of per-pixel errors does not.

**Analytic point components.** A point of amplitude $a$ at $\mathbf x_0$ has visibilities $a\,e^{-2\pi i\,\mathbf u\cdot\mathbf x_0}$, again linear in $a$, so it joins the same equation as extra rows and columns:

$$
\begin{pmatrix}F+H & F_{sa}\\ F_{sa}^T & F_{aa}\end{pmatrix}
\begin{pmatrix}s\\ a\end{pmatrix} =
\begin{pmatrix}D\\ D_a\end{pmatrix},
\quad
(F_{sa})_{i} = \Sigma w\,(M^T B(\cdot-\mathbf x_0))_i,\;
F_{aa} = \Sigma w,\;
D_a = \Sigma w\,I'(\mathbf x_0).
$$

These are the same $B$ and $I'$, evaluated off the grid. In streaming they are read by smooth interpolation from grids 8× finer than the image.

**Cubes.** The uv coordinates scale with frequency, so each channel has its own $\tilde W$, $I'$ and $c$. All channels share one prior ($\lambda$, the correlation length and an adaptive brightness map) fixed by the MFS fit.

**Noise.** $w_k$ sets $\chi^2$, $\rho$, the criteria and every uncertainty, and the measurement-set weights are only relative. So $\sigma_k$ is estimated from the data by differencing visibilities adjacent in time on the same baseline: $\sigma = {\rm std}(\Delta V)/\sqrt2$.
