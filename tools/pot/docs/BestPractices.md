#  Post-Training Quantization Best Practices {#pot_docs_BestPractices}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   Saturation Issue <pot_saturation_issue>

@endsphinxdirective

The [Default Quantization](@ref pot_default_quantization_usage) of the Post-training Optimization Tool (POT) is 
the fastest and easiest way to get a quantized model. It requires only some unannotated representative dataset to be provided in most cases. Therefore, it is recommended to use it as a starting point when it comes to model optimization. However, it can lead to significant accuracy deviation in some cases. The purpose of this article is to provide tips to address this issue.

> **NOTE**: POT uses inference on the CPU during model optimization. It means that ability to infer the original
> floating-point model is essential for model optimization. 
> It is also worth mentioning that in case of the 8-bit quantization, it is recommended to run POT on the same CPU
> architecture when optimizing for CPU or VNNI-based CPU when quantizing for a non-CPU device, such as GPU, VPU, or GNA.
> It should help to avoid the impact of the [saturation issue](@ref pot_saturation_issue) that occurs on AVX and SSE based CPU devices. 

## Improving accuracy after the Default Quantization
Parameters of the Default Quantization algorithm with basic settings are presented below:
```python
{
    "name": "DefaultQuantization", # Optimization algorithm name
    "params": {
        "preset": "performance", # Preset [performance, mixed] which controls 
                                 # the quantization scheme. For the CPU: 
                                 # performance - symmetric quantization  of weights and activations.
                                 # mixed - symmetric weights and asymmetric activations.
        "stat_subset_size": 300  # Size of subset to calculate activations statistics that can be used
                                 # for quantization parameters calculation.
    }
}
```

There are two alternatives in case of substantial accuracy degradation after applying this method:
1.  Hyperparameters tuning.
2.  AccuracyAwareQuantization algorithm.

### Tuning Hyperparameters of the Default Quantization
The Default Quantization algorithm provides multiple hyperparameters which can be used in order to improve accuracy results for the fully-quantized model. 
Below is a list of best practices that can be applied to improve accuracy without a substantial performance reduction with respect to default settings:
1.  The first recommended option is to change the `preset` from `performance` to `mixed`. This enables asymmetric quantization of 
activations and can be helpful for models with non-ReLU activation functions, for example, YOLO, EfficientNet, etc.
2.  The second option is the `use_fast_bias`. Setting this option to `false` enables a different bias correction method which is generally more accurate
and applied after model quantization, as a part of the Default Quantization algorithm.
   > **NOTE**: Changing this option can substantially increase quantization time in POT tool.
3.  Another important option is the `range_estimator`. It defines how to calculate the minimum and maximum of quantization range for weights and activations.
For example, the following `range_estimator` for activations can improve the accuracy for Faster R-CNN based networks:
```python
{
    "name": "DefaultQuantization", 
    "params": {
        "preset": "performance", 
        "stat_subset_size": 300  
                                    

        "activations": {                 # defines activation
            "range_estimator": {         # defines how to estimate statistics 
                "max": {                 # right border of the quantizating floating-point range
                    "aggregator": "max", # use max(x) to aggregate statistics over calibration dataset
                    "type": "abs_max"    # use abs(max(x)) to get per-sample statistics
                }
            }
        }
    }
}
```

Find the possible options and their description in the `configs/default_quantization_spec.json` file in the POT directory.

4.  The next option is the `stat_subset_size`. It controls the size of the calibration dataset used by POT to collect statistics for quantization parameters initialization.
It is assumed that this dataset should contain a sufficient number of representative samples. Hence, varying this parameter may affect accuracy (higher is better). 
However, it proves that 300 samples are sufficient to get representative statistics in most cases.
5.  The last option is the `ignored_scope`. It allows excluding some layers from the quantization process, for example, their inputs will not be quantized. It may be helpful for some patterns, which are known in advance, that they drop accuracy when executing in low-precision.
For example, the `DetectionOutput` layer of SSD model expressed as a subgraph should not be quantized to preserve the accuracy of Object Detection models.
One of the sources for the ignored scope can be the Accuracy-aware algorithm, which can revert layers back to the original precision (see the details below).

## Accuracy-aware Quantization
If the steps above do not result in an accurate quantized model, you may use the so-called [Accuracy-aware Quantization](@ref pot_accuracyaware_usage) algorithm, which produces mixed-precision models. 
Here is a fragment of Accuracy-aware Quantization configuration with default settings:
```python
{
    "name": "AccuracyAwareQuantization",
    "params": {
        "preset": "performance", 
        "stat_subset_size": 300,

        "maximal_drop": 0.01 # Maximum accuracy drop which has to be achieved after the quantization
    }
}

```

Since the Accuracy-aware Quantization calls the Default Quantization at the first step it means that all the parameters of the latter one are also valid and can be applied to the accuracy-aware scenario.

> **NOTE**: In general, the potential increase in speed with the Accuracy-aware Quantization algorithm is not as high  as with the Default Quantization, when the model gets fully quantized.

### Reducing the performance gap of Accuracy-aware Quantization
To improve model performance after Accuracy-aware Quantization, try the `"tune_hyperparams"` setting and set it to `True`. It will enable searching for optimal quantization parameters before reverting layers to the "backup" precision. Note that this may impact the overall quantization time, though.

If the Accuracy-aware Quantization algorithm does not provide the desired accuracy and performance or you need an accurate, fully-quantized model, use [NNCF](@ref docs_nncf_introduction) for Quantization-Aware Training.
