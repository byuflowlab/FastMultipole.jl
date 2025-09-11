# Theory

In a nutshell, `FastMultipole` operates by forming series expansions of a kernel function, and translating and combining those expansions to obtain optimal compression. You can get a sense for how this works in the following figure.

```@raw html
<img src="../assets/multipole_expansions.png" width="660px"/>
```

Here, we see how each body's influence can be expressed as a series expansion. These series expansions can be translated and combined such that an entire cluster of bodies is represented by a single series expansion. This is what is known as a multipole expansion. Multipole expansions only converge outside of a finite radius of convergence, as illustrated by the red dotted line. Multipole expansions can only be used for interactions that are farther apart than this circl. The accuracy of the expansion gets better and better the farther away we go, so we can control the accuracy by imposing a cutoff radius (dotted blue line), and only use multipole expansions for interactions that are farther away than the blue circle. Multipole expansions are very helpful for reducing the cost of the N-body problem; in fact, we can reduce the scaling of the N-body problem to O(NlogN) by only considering multipole expansions.

Local expansions are very similar to multipole expansions, but they converge inside of a finite radius rather than outside. These provide the additional required compression to achieve fully O(N) scaling. In the next figure, we see how local expansions can reduce the number of times an expansion need be evaluated.

```@raw html
<img src="../assets/local_expansions.png" width="660px"/>
```

More details of the fast multipole method (FMM) can be found in the original work by Greengard and Rokhlin.[greengard1987fast](@cite)

## References

```@bibliography
```
