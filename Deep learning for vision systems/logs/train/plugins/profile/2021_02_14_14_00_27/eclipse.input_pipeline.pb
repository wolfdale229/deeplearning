	���GHG@���GHG@!���GHG@	y�P���?y�P���?!y�P���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���GHG@�,�"�J�?A��W�G@Y� �X�?*	�E����k@2F
Iterator::Model��j̹?!�#�ԐF@)�ֈ`\�?1��R/�7@:Preprocessing2U
Iterator::Model::ParallelMapV2�"M�<�?!p����6@)�"M�<�?1p����6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���L0��?!�j�1�&8@)8��w��?18p��/4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����?�?!u�ZY2V4@)�z�G�?10�y)R�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��ӝ'��?!��;�)!@)��ӝ'��?1��;�)!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-wf��\�?!x�m�+oK@)���aN�?1�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorѕT� �?!ӷ�Q�@)ѕT� �?1ӷ�Q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����a��?!r�Ǖ7@)5&�\R�}?1�?���	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9y�P���?Is�}�&�X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�,�"�J�?�,�"�J�?!�,�"�J�?      ��!       "      ��!       *      ��!       2	��W�G@��W�G@!��W�G@:      ��!       B      ��!       J	� �X�?� �X�?!� �X�?R      ��!       Z	� �X�?� �X�?!� �X�?b      ��!       JCPU_ONLYYy�P���?b qs�}�&�X@