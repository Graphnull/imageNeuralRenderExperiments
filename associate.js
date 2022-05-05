let tf = require('@tensorflow/tfjs-node')

class Associate {
  /**
   *  @type {tf.TensorBuffer}
   */
  _clusters
  /**
   * @type {tf.Tensor}
   */
  clusters = []
  inputs = 0
  size = 0;
  /**
   * 
   * @param {number} size 
   * @param {number} inputs 
   */
  constructor(size, inputs) {
    this.inputs = inputs
    this.size = size
    this.clusters = tf.stack(new Array(size).fill(tf.zeros([inputs])))
    this._clusters = this.clusters.bufferSync()
    this.clustersCount = tf.tensor(new Array(size).fill(0), [size]).bufferSync()
  }
  /**
   * 
   * @param {tf.Tensor} arr 
   * @param {tf.Tensor} outarr 
   */
  push(arr, outarr) {
    if ((arr.shape.length || 1) !== 1) {

      throw new Error('arr shape length !== 1')
    }
    if ((arr.shape[0] || 1) !== this.inputs) {
      throw new Error('arr.shape[0]!==this.inputs')
    }
    tf.tidy(() => {


      let minI
      if (outarr) {
        minI = outarr.squaredDifference(tf.stack(this.clustersOut)).sum(-1).sqrt().mul(this.clustersCount.toTensor()).argMin().dataSync()[0]
      } else {
        minI = arr.squaredDifference(this.clusters).sum(-1).sqrt().mul(this.clustersCount.toTensor()).argMin().dataSync()[0]
      }

      let count = this.clustersCount.get(minI)
      let newCluster = this.clusters.slice([minI, 0], [1, this.inputs]).mul(count / (count + 1)).add(arr.mul(1 / (count + 1)))

      let cl = this._clusters
      cl.values.set(newCluster.dataSync(), minI * this.inputs)
      this.clusters = cl.toTensor();
      tf.keep(this.clusters)
      this.pushAdd && this.pushAdd(minI, outarr)

      this.clustersCount.set(count + 1, minI);
    })
  }
  /**
   * 
   * @param {tf.Tensor} arr 
   * @param {tf.Tensor} mask
   */
  reversePredict(arr, mask) {
    if ((arr.shape.length || 1) !== 1) {
      throw new Error('arr.shape.length!==1')
    }
    if ((arr.shape[0] || 1) !== this.inputs) {
      throw new Error('arr.shape[0]!==this.inputs')
    }
    return tf.tidy(() => {

      let distances;
      if (mask) {
        distances = arr.squaredDifference(this.clusters).mul(mask).sum(-1).sqrt()
      } else {
        distances = arr.squaredDifference(this.clusters).sum(-1).sqrt()
      }
      //let max = distances.max().sub(0.000001)
      //let min = distances.min().sub(0.000001);
      //distances = distances.sub(min)
      let sum = distances.sum().add(0.00001)
      let mixes = sum.div(distances).minimum(2 ** 22).softmax()
      //console.log('mixes: ', mixes.dataSync(), distances.dataSync(), arr.dataSync(), this.inputs);

      //console.log('reversePredict: ', tf.stack(this.clusters), mixes);
      let res = { inputs: this.clusters.mul(mixes.reshape([this.size, 1])).sum(-2), mixes }
      this.reversePredictAdd && this.reversePredictAdd(res, mixes)

      return res
    })
  }
}
class AssociateWithOutput extends Associate {
  /**
   * @type {tf.Tensor[]}
   */
  clustersOut = []

  outputs = 0;
  /**
   * 
   * @param {number} size 
   * @param {number} inputs 
   * @param {number} outputs 
   */
  constructor(size, inputs, outputs) {
    super(size, inputs);
    this.outputs = outputs
    this.clustersOut = new Array(size).fill(new tf.zeros([outputs]))
  }
  /**
   * 
   * @param {number} index 
   */
  pushAdd(index, outarr) {
    let count = this.clustersCount.get(index)
    let newCluster = this.clustersOut[index].mul(count / (count + 1)).add(outarr.mul(1 / (count + 1)))

    //tf.dispose(this.clustersOut[index])
    tf.keep(newCluster)
    this.clustersOut[index] = newCluster;
  }
  reversePredictAdd(res, mixes) {
    res.outputs = tf.stack(this.clustersOut).mul(mixes.reshape([this.size, 1])).sum(-1).reshape([1])

  }
}
module.exports.Associate = Associate
module.exports.AssociateWithOutput = AssociateWithOutput