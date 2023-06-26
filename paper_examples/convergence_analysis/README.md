To plot the exact solution:

- Run `viz_exact_solution.py` with the preferred `mesh_size`.

To perform the convergence analysis:

- Run `run_analysis.py`. This will create a pickle file named `l2_errors.pickle` 
  inside the `out` folder.
- Now, to plot the rates, simply run `viz_l2_errors.py`.