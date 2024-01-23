let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');



let w = 480;
let h = 352;

let psize = 128;
let save = async (tensor, name) => {
  let bin = await tf.node.encodeJpeg(tensor.minimum(255).maximum(0), "rgb", 100, false, true, false)
  await fs.promises.writeFile(name, bin);
}

let xys = [
  [0, 0], [25, 35], [45, 75], [110, 130], [185, 160], [223, 180], [223, 210], [223, 240], [223, 270], [223, 300], [223, 330], [223, 350],
  [205, 350], [165, 350], [125, 350], [105, 350],
  [115, 330], [125, 290], [135, 260],
  [115, 250], [65, 230], [35, 250], [35, 210], [35, 180],
]
let modeltest = async (origImages, xyWeights) => {

  let back = tf.tidy(() => {
    let th = h * 1.5
    let tw = w * 1.5
    let back = tf.zeros([th, tw, 3])

    origImages.forEach((img, i) => {
      let x = Math.round(xyWeights[i * 2 + 1]) + 80
      let y = Math.round(xyWeights[i * 2 + 0]) + 80

      let m = tf.zeros([psize, psize, 3]).pad([
        [Math.min(Math.max(0, x), th - psize), Math.max(0, Math.min((th - psize) - x, th - psize))],
        [Math.min(Math.max(0, y), tw - psize), Math.max(0, Math.min((tw - psize) - y, tw - psize))],
        [0, 0]], 1)

      back = back.mul(m).maximum(img.reshape([psize, psize, 3]).pad([
        [Math.min(Math.max(0, x), th - psize), Math.max(0, Math.min((th - psize) - x, th - psize))],
        [Math.min(Math.max(0, y), tw - psize), Math.max(0, Math.min((tw - psize) - y, tw - psize))],
        [0, 0]]))
    })
    return back.mul(255)
  })
  await save(back, './temp/result.jpg')
  back.dispose()
}

/**
 * 
 * @param {tf.Tensor} imgg 
 * @param {number} count 
 * @returns {[tf.Tensor, {x: number;y: number;v0: number;v1: number;v2: number;v3: number;v4: number;}[]]}
 */
let fdetect = (imgg, count) => {
  let img = tf.conv2d(imgg, tf.concat(new Array(imgg.shape[2]).fill(0).map(v => tf.tensor([0.5, 1, 0.5, 1, -6, 1, 0.5, 1, 0.5], [3, 3, 1, 1])), -1).reshape([3, 3, imgg.shape[2], 1]), 1, 'valid').relu()

  let w = img.shape[1]
  let h = img.shape[0]
  let data = img.dataSync();
  let ndata = new Float32Array((w - 2) * (h - 2) * 5)
  for (let x = 1; x < (w - 1); x++) {
    for (let y = 1; y < (h - 1); y++) {
      let k = data[(y - 0) * w + (x - 0)]
      let pos = (y * (w - 2)) * 5 + x * 5
      ndata[pos + 0] = Math.abs(data[(y - 1) * w + (x - 0)] - data[(y + 1) * w + (x - 0)]) * k
      ndata[pos + 1] = Math.abs(data[(y - 1) * w + (x - 1)] - data[(y + 1) * w + (x + 1)]) * k
      ndata[pos + 2] = Math.abs(data[(y - 0) * w + (x - 1)] - data[(y + 0) * w + (x + 1)]) * k
      ndata[pos + 3] = Math.abs(data[(y + 1) * w + (x - 1)] - data[(y - 1) * w + (x + 1)]) * k
      ndata[pos + 4] = (ndata[pos + 0] + ndata[pos + 1] + ndata[pos + 2] + ndata[pos + 3]) / 4

    }
  }
  let vecimg = tf.tensor(ndata, [h - 2, w - 2, 5])//.pool([2, 2], 'avg', 'valid', 1, 2)
  img = vecimg.slice([0, 0, 4], [vecimg.shape[0], vecimg.shape[1], 1])


  let descs = new Array(vecimg.shape[0] * vecimg.shape[1]).fill(0).map(v => ({ x: 0, y: 0, v0: 0, v1: 0, v2: 0, v3: 0, v4: 0 }))
  let vecdata = vecimg.dataSync()
  for (let i = 0; i < (vecimg.shape[0] * vecimg.shape[1]); i++) {
    let x = i % vecimg.shape[1]
    let y = Math.round(i / vecimg.shape[1])
    descs[i] = { x, y, v0: vecdata[i * 5 + 0], v1: vecdata[i * 5 + 1], v2: vecdata[i * 5 + 2], v3: vecdata[i * 5 + 3], v4: vecdata[i * 5 + 4] }
  }

  descs = descs.map((v, i, arr) => (arr[i * 2] && arr[i * 2].v4) > (arr[i * 2 + 1] && arr[i * 2 + 1].v4) ? (arr[i * 2] && arr[i * 2]) : (arr[i * 2 + 1] && arr[i * 2 + 1])).slice(0, Math.floor(descs.length / 2))
  return [vecimg, descs.sort((l, r) => r.v4 - l.v4).slice(0, count)]
};

; (async () => {




  let img = tf.node.decodeJpeg(await fs.promises.readFile('./im.jpg')).slice([0, 0, 0], [h, w, 3]).toFloat().div(255)

  let origImages = xys.map(xy => img.slice(xy, [psize, psize]).add(tf.randomNormal([psize, psize, 3]).mul(0.02)));

  let descriptors = origImages.map(img => {

    return fdetect(img, 100)[1]

  })
  let avgDiff = (minI) => {
    let iters = minI.length
    let avgx = 0
    let avgy = 0
    for (let iter = 0; iter < iters; iter++) {

      avgx += minI[iter].x1 - minI[iter].x2
      avgy += minI[iter].y1 - minI[iter].y2
    }
    return [avgx / iters, avgy / iters]
  }


  /**
   * 
   * @param {ReturnType<fdetect>[1]} ld 
   * @param {ReturnType<fdetect>[1]} rd 
   * @returns 
   */
  let getСonformity = (ld, rd, dx = 0, dy = 0) => {
    let ldescriptor = ld.filter(v => v.x >= Math.max(0, dx) && v.y >= Math.max(0, dy) && v.x < Math.min(psize, psize + dx) && v.y < Math.min(psize, psize + dy))
    let rdescriptor = rd.filter(v => (v.x + dx) < psize && (v.x + dx) >= 0 && (v.y + dy) < psize && (v.y + dy) >= 0)
    if (!rdescriptor.length || !ldescriptor.length) {
      return [];
    }

    let minI = []
    let iters = Math.min(ldescriptor.length - 1, rdescriptor.length - 1, 32)
    let lastm = 0
    for (let iter = 0; iter < iters; iter++) {
      let min = Infinity
      for (let i = 0; i < ldescriptor.length; i++) {
        for (let j = 0; j < rdescriptor.length; j++) {

          let m = 0
          m += Math.pow(ldescriptor[i].v0 - rdescriptor[j].v0, 2)
          m += Math.pow(ldescriptor[i].v1 - rdescriptor[j].v1, 2)
          m += Math.pow(ldescriptor[i].v2 - rdescriptor[j].v2, 2)
          m += Math.pow(ldescriptor[i].v3 - rdescriptor[j].v3, 2)
          if (isNaN(m)) {
            console.log(m);
          }
          lastm = m
          if (min > m) {
            min = m
            minI[iter] = { i, j, x1: ldescriptor[i].x, y1: ldescriptor[i].y, x2: rdescriptor[j].x, y2: rdescriptor[j].y }
          }
        }
      }

      try {
        ldescriptor.splice(minI[iter].i, 1)
        rdescriptor.splice(minI[iter].j, 1)
      } catch (err) {
        console.log(ldescriptor.length, rdescriptor.length, lastm, minI.length, min, iter);
        throw err;
      }

    }
    let [avgx, avgy] = avgDiff(minI)
    console.log('avgx, avgy: ', avgx, avgy);


    minI = minI.filter(v => {

      return (v.x2 + avgx) < psize && (v.x2 + avgx) >= 0 && (v.y2 + avgy) < psize && (v.y2 + avgy) >= 0 &&
        v.x1 >= Math.max(0, avgx) && v.y1 >= Math.max(0, avgy) && v.x1 < Math.min(psize, psize + avgx) && v.y1 < Math.min(psize, psize + avgy)
    })

    avgx = avgDiff(minI)[0]
    avgy = avgDiff(minI)[1]
    minI = minI.filter(v => {
      return Math.sqrt(Math.pow((v.x1 - v.x2) - avgx, 2) + Math.pow((v.y1 - v.y2) - avgy, 2)) < (psize * 0.2)
    })

    avgx = avgDiff(minI)[0]
    avgy = avgDiff(minI)[1]

    console.log('avgx, avgy11: ', avgx, avgy);
    minI = minI.filter(v => {

      return (v.x2 + avgx) < psize && (v.x2 + avgx) >= 0 && (v.y2 + avgy) < psize && (v.y2 + avgy) >= 0 &&
        v.x1 >= Math.max(0, avgx) && v.y1 >= Math.max(0, avgy) && v.x1 < Math.min(psize, psize + avgx) && v.y1 < Math.min(psize, psize + avgy)
    })

    avgx = avgDiff(minI)[0]
    avgy = avgDiff(minI)[1]
    minI = minI.filter(v => {
      return Math.sqrt(Math.pow((v.x1 - v.x2) - avgx, 2) + Math.pow((v.y1 - v.y2) - avgy, 2)) < (psize * 0.2)
    })

    avgx = avgDiff(minI)[0]
    avgy = avgDiff(minI)[1]

    minI = minI.filter(v => {

      return (v.x2 + avgx) < psize && (v.x2 + avgx) >= 0 && (v.y2 + avgy) < psize && (v.y2 + avgy) >= 0 &&
        v.x1 >= Math.max(0, avgx) && v.y1 >= Math.max(0, avgy) && v.x1 < Math.min(psize, psize + avgx) && v.y1 < Math.min(psize, psize + avgy)
    })

    avgx = avgDiff(minI)[0]
    avgy = avgDiff(minI)[1]
    console.log('avgx, avgy222: ', avgx, avgy);

    //if (second) {console.log('minI: ', minI);
    return minI

    //} else {
    // return getСonformity(ldescriptor, rdescriptor, true)
    //}
  }
  let visualize = async (t, minI, p, path) => {
    let tdata = t.dataSync()
    let iters = minI.length
    for (let iter = 0; iter < iters; iter++) {
      tdata[minI[iter]['y' + p] * t.shape[0] * t.shape[2] + minI[iter]['x' + p] * t.shape[2] + 0] = 1;
      tdata[minI[iter]['y' + p] * t.shape[0] * t.shape[2] + minI[iter]['x' + p] * t.shape[2] + 1] = 0;
      tdata[minI[iter]['y' + p] * t.shape[0] * t.shape[2] + minI[iter]['x' + p] * t.shape[2] + 2] = 1;
    }
    await save(tf.tensor(tdata, t.shape).mul(255), './temp/' + path + '.jpg')
  }


  let pairDescs = [];
  for (let i = 1; i < descriptors.length; i++) {

    pairDescs.push(getСonformity(descriptors[i - 1], descriptors[i]))
  }

  let pairDescsSec = [];

  for (let i = 2; i < descriptors.length; i++) {
    //console.log('pairDescs[i - 1]: ', pairDescs[i - 1]);
    let [avgx1, avgy1] = avgDiff(pairDescs[i - 2])

    let [avgx2, avgy2] = avgDiff(pairDescs[i - 1])


    //console.log(avgx1 + avgx2, avgy1 + avgy2);
    pairDescsSec.push(getСonformity(descriptors[i - 2], descriptors[i], avgx1 + avgx2, avgy1 + avgy2))
  }


  await visualize(origImages[0], pairDescs[0], '1', 'tt')
  await visualize(origImages[1], pairDescs[0], '2', 'tt2')



  let back = tf.zeros(img.shape)
  origImages.forEach((img, i) => {
    let m = tf.zeros([psize, psize, 3]).pad([[xys[i][0], (h - psize) - xys[i][0]], [xys[i][1], (w - psize) - xys[i][1]], [0, 0]], 1)
    back = back.mul(m).maximum(img.reshape([psize, psize, 3]).pad([[xys[i][0], (h - psize) - xys[i][0]], [xys[i][1], (w - psize) - xys[i][1]], [0, 0]]))
  })
  await save(back.mul(255), './temp/back.jpg')

  let xyWeights = new Float32Array(origImages.length * 2)

  let l1pos = [0, 0]
  let l2pos = [0, 0]
  let error = 0
  //console.log('pairDescs: ', pairDescs[2], pairDescsSec[1]);
  for (let i = 0; i < pairDescs.length; i++) {

    let [avgx, avgy] = avgDiff(pairDescs[i])

    xyWeights[i * 2 + 2] = avgx + l1pos[0];
    xyWeights[i * 2 + 3] = avgy + l1pos[1];
    if (pairDescsSec[i - 1] && pairDescsSec[i - 1].length) {

      let [avgxl, avgyl] = avgDiff(pairDescsSec[i - 1])
      console.log('pos', i, avgxl + l2pos[0], avgx + l1pos[0], xys[i + 1][1], pairDescs[i].length, pairDescsSec[i - 1].length);
      let k = pairDescs[i].length / (pairDescs[i].length + pairDescsSec[i - 1].length);
      //xyWeights[i * 2 + 2] = (avgx + l1pos[0]) * (k) + (avgxl + l2pos[0]) * (1 - k);
      //xyWeights[i * 2 + 3] = (avgy + l1pos[1]) * (k) + (avgyl + l2pos[1]) * (1 - k);
      //console.log('avgx, avgyllllll: ', i, avgxl, avgyl);
    }
    l2pos = l1pos;
    l1pos = [xyWeights[i * 2 + 2], xyWeights[i * 2 + 3]]
    let err = Math.sqrt(Math.pow(xys[i + 1][1] - xyWeights[i * 2 + 2], 2) + Math.pow(xys[i + 1][0] - xyWeights[i * 2 + 3], 2))
    if (err > 100) {
      console.log('e1', i, err);
    }
    error += err

  }
  console.log('error', error / pairDescs.length)

  xyWeights = tf.tensor(xyWeights, [origImages.length, 2])
  //console.log('xyWeights: ', xyWeights);
  await modeltest(origImages, xyWeights.dataSync())
  //console.log('xyWeights: ', xyWeights.dataSync());

})();