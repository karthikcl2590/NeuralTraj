Matlab code for extracting latent variables using 
Time-delay Gaussian-process factor analysis (TD-GPFA).

Version 1.01        May 1, 2017

The TD-GPFA algorithm is described in detail in the following 
reference. Please read the paper carefully before using the code, 
as it describes all of the terminology and usage modes:

"Extracting Low-Dimensional Latent Structure from Time Series 
in the Presence of Delays"
by K. C. Lakshmanan, P. T. Sadtler, E. C. Tyler-Kabara, A. P. Batista, B. M. Yu.
Neural Computation 2015

==========
COPYRIGHT
==========
@ 2015 Karthik Lakshmanan     karthikl@cs.cmu.edu
       Byron Yu               byronyu@cmu.edu
       
This code may not be distributed in any form without prior consent
of the authors.  While it has been carefully checked for bugs and
applied to many datasets, the code comes with no warranty of any
kind.  The authors take no responsibility for support or maintenance
of the code, although feedback is appreciated.

===============
VERSION HISTORY
===============
In notes below, + denotes new feature, - denotes bug fix or removal.

Version 1.01     — April 2015

====================
HOW TO GET STARTED
====================

Look at example_tdgpfa.m

Note: Matlab should be started in the current directory (NeuralTraj) so
that it executes startup.m automatically.

==================================================
ARE THERE KNOBS I NEED TO TURN WHEN FITTING TD-GPFA?
==================================================

1) Spike bin width (‘binWidth’)
   See corresponding section in README_GPFA.

2) Initialization option (‘init_method’)
   The ECME algorithm for TD-GPFA can be initialized using two methods:
   ‘MSISA’ - 	parameters are initialized using Multitrial Shift Invariant
		Subspace Analysis, described in the reference above. This is the 
		default option.
   ‘GPFA’  - 	parameters are initialized by using parameters estimated by GPFA
		on the same dataset with the same latent dimensionality. The initial 
		values for the delays are all set to 0. For some datasets, this results
		in estimated models with higher cross-validated likelihood than when 
		initialized with MSISA.

3) Maximum Delay (‘maxDelayFrac’)
   This allows the user to bound the largest values the estimated delays can take,
   as described in the Methods section of the reference above. ‘maxDelayFrac’ is set 
   by default to 0.5, which means the magnitude of the largest allowed delay is 0.5 
   times the length of the shortest trial. 

==================================================
HOW CAN I MAKE THE CODE RUN FASTER?
==================================================

TD-GPFA model fitting can be slow. (See Appendix C in reference above). Here are some tips
that can help reduce computational time.

1) IMPORTANT: We highly recommend that the number of unique trial lengths in the dataset
   is as small as possible. Ideally, every trial in the dataset is of the same length. This 
   results in massive speed up because large parts of the computation can be performed
   once per unique trial length.

2) Use fewer ECME iterations (dangerous)

   Tradeoff: ECME might not have converged yet.
  
   How to do it: Set 'ecmeMaxIters', an optional argument to neuralTraj.m.

   Running time scales linearly with 'ecmeMaxIters'.

3) Use a smaller state dimensionality

   Tradeoff: Might not be able to capture all the structure in data.
   
   How to do it: Set 'xDim', an optional argument to neuralTraj.m.

   Running time scales roughly with xDim^3.

4) For cross-validation, use the parallelization option.
   
   How to do it: Set ‘parallelize’, an optional argument to neuralTraj.m.

5) NOTE: Because latent variables can drive neurons after arbitrarily large delays, 
   TD-GPFA cannot make use of cutting trials into shorter segments, so setting 
   ‘segLength’ (as can be done for GPFA) provides no benefit.

=============================
NOTE ON THE USE OF C/MEX CODE
=============================

The function grad_DelayMatrix_LL_constrained.m uses invChol_mex() to speed up an expensive 
symmetric matrix inversion by exploiting symmetry in a C environment.

For more information on using MEX, read the corresponding section in README_GPFA.

================================================
WHAT DOES THE WARNING ABOUT PRIVATE NOISE MEAN?
================================================

Read corresponding section in README_GPFA.



