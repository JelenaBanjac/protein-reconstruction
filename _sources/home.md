# Learning to Recover Orientations from Projections in Single-Particle Cryo-EM
 
[Jelena Banjac](https://jelenabanjac.com), jelena.banjac@epfl.ch, Data Science Master Student  
[Laurène Donati](https://people.epfl.ch/laurene.donati?lang=en), laurene.donati@epfl.ch, BIG, EPFL  
[Michaël Defferrard](https://deff.ch/), michael.defferrard@epfl.ch, LTS2, EPFL  

## Summary
A major challenge in single-particle cryo-electron microscopy (cryo-EM) is that the orientationsadopted by the 3D particles prior to imaging are unknown;  yet, this knowledge is essential forhigh-resolution reconstruction. We present a method to recover these orientations directly from theacquired set of 2D projections. Our approach consists of two steps: (i) the estimation of distancesbetween pairs of projections, and (ii) the recovery of the orientation of each projection from thesedistances.  In step (i), pairwise distances are estimated by a Siamese neural network trained onsynthetic cryo-EM projections from resolved bio-structures. In step (ii), orientations are recovered byminimizing the difference between the distances estimated from the projections and the distancesinduced by the recovered orientations.  We evaluated the method on synthetic cryo-EM datasets.Current results demonstrate that orientations can be accurately recovered from projections that areshifted and corrupted with a high level of noise.  The accuracy of the recovery depends on theaccuracy of the distance estimator. While not yet deployed in a real experimental setup, the proposedmethod offers a novel learning-based take on orientation recovery in SPA and may bring interestingnew perspectives in the field.  Our code is available at https://github.com/JelenaBanjac/protein-reconstruction.

## Two-Step Method
Our method consists of two steps.  First, we estimate distances between pairs of projections.  Second, werecover the orientation of each projection from these distances.
![images/protein_flow.png](images/schematic_method_overview-1.jpg)


