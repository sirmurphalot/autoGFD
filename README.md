<div align="center">
<img src="man/figures/autoGFD.png" alt="logo" width=75%></img>
</div>
<br>

# **Autodifferentiation for Generalized Fiducial Inference** 
With the recent compatibility between JAX and TensorFlow Probability, we thought it possible to create a user-friendly autodifferentiator for any Generalized Fiducial application.  

<ins>NEWS 11/09/2021</ins>: We have had moderate success with simple examples, but have hit snags with scaling this up to higher dimensions, such as with the MVN model, and with overly complex derivatives, such as with the Gamma model.


## Examples

To run the examples, you must run the scripts as modules.  For instance, to run the `simple_normal` example, use the following command line prompt in the autoGFD folder.

```console
foo@bar autoGFD$ python -m examples.simple_normal
```

## Packages Required

`tensorflow_probability`, `jax`, `warnings`, `seaborn`, `pandas`, `time`, `scipy`, `numpy`, `matplotlib`, `os`

## Citation

None.

