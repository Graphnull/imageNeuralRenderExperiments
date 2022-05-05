let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');
let associate = require('./associate')
let datalen = 250;
let motion = tf.tensor(JSON.parse(fs.readFileSync('./motion.json')).slice(100, datalen + 100).map(v => v['gravity.y']), [datalen, 1]);

let clamp = (v, min, max) => Math.max(Math.min(v, max), min)

/**
 * 
 * @param {tf.Tensor} tensor 
 */
let visualize = (tensor) => {
  tf.tidy(() => {
    let data = tensor.dataSync()
    let vis = new Array(30).fill(0).map(v => new Array(data.length).fill(' '))
    data.forEach((v, i) => {
      vis[clamp(((v - 0.85) * 175) | 0, 0, 29)][i] = '0'
    })
    vis.map(v => console.log(v.join('')))
  })
}

console.log('data:');
//visualize(motion)




let xys = [5, 40, 50, 60, 75, 100, 120, 200, 215];


let uvdata = new Float32Array(datalen)
for (let x = 0; x < datalen; x++) {
  uvdata[x] = x / datalen
}


let fuv = tf.tensor(uvdata, [datalen, 1])
// uv = fuv.slice([0, 0], [96, 96])

// let fuv1 =fuv.slice([0, 0, 0], [h, w, 1])
// let fuv2 =fuv.slice([0, 0, 1], [h, w, 1])
let random = tf.concat([
  //fuv,
  fuv.mul(datalen * Math.PI / 1.5 + 0.8160475840461483).cos(),
  fuv.mul(datalen * Math.PI / 1 + 0.8581418410314106).cos(),
  fuv.mul(datalen * Math.PI / 2 + 0.2577779787891872).cos(),
  fuv.mul(datalen * Math.PI / 3 + 0.09797944575707351).cos(),
  fuv.mul(datalen * Math.PI / 4 + 0.19792368286245177).cos(),
  fuv.mul(datalen * Math.PI / 6 + 0.949482165305719).cos(),
  fuv.mul(datalen * Math.PI / 8 + 0.8687709068294289).cos(),
  fuv.mul(datalen * Math.PI / 10 + 0.615134161356331).cos(),
  fuv.mul(datalen * Math.PI / 12 + 0.8411972620860926).cos(),
  fuv.mul(datalen * Math.PI / 13 + 0.07408877615899767).cos(),
  fuv.mul(datalen * Math.PI / 16 + 0.8153963784798905).cos(),
  fuv.mul(datalen * Math.PI / 37 + 0.44152425770235726).cos(),
  fuv.mul(datalen * Math.PI / 67 + 0.20640195692894525).cos(),
  fuv.mul(datalen * Math.PI / 120 + 0.6808663003808029).cos(),
], -1)
//console.log(fuv.mul(datalen * Math.PI / 2).cos().dataSync())

let inputLen = random.shape[random.shape.length - 1]

let mem = new associate.AssociateWithOutput(1, inputLen, 1)
let detect = new associate.AssociateWithOutput(1, 3, 1)
let mem2 = new associate.AssociateWithOutput(1, inputLen + 1, 1)

let modelPrep = () => {
  const model = tf.sequential();

  model.add(tf.layers.dense({ units: 64, activation: 'linear', inputShape: [inputLen] }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  model.add(tf.layers.dense({ units: 48, activation: 'linear', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  //model.add(tf.layers.dense({ units: 40, activation: 'linear', }));
  //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  //model.add(tf.layers.batchNormalization({}));
  //model.add(tf.layers.dense({ units: 32, activation: 'linear', }));
  //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  model.add(tf.layers.dense({ units: 24, activation: 'linear', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  model.add(tf.layers.dense({ units: 16, activation: 'linear', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  model.add(tf.layers.dense({ units: 12, activation: 'linear', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  //model.add(tf.layers.batchNormalization({}));

  model.add(tf.layers.dense({ units: 8, activation: 'linear', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  //model.add(tf.layers.batchNormalization({}));
  model.add(tf.layers.dense({ units: 1, activation: 'linear', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: 'meanSquaredError', optimizer: tf.train.adam(0.0004) });
  //console.log('model: ', model.summary());

  return model;
}
let modelPrepTest = (low) => {

  const model = tf.sequential();

  model.add(tf.layers.conv1d({ filters: 24, kernelSize: [3], padding: 'same', activation: 'linear', inputShape: [null, 1] }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  model.add(tf.layers.conv1d({ filters: 16, kernelSize: [1], padding: 'same', activation: 'linear', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  !low && model.add(tf.layers.conv1d({ filters: 12, kernelSize: [3], padding: 'same', activation: 'linear', }));
  !low && model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  !low && model.add(tf.layers.conv1d({ filters: 8, kernelSize: [1], padding: 'same', activation: 'linear', }));
  !low && model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  model.add(tf.layers.conv1d({ filters: 1, kernelSize: [1], padding: 'same', activation: 'linear', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: 'meanSquaredError', optimizer: tf.train.adam(0.0004) });
  //console.log('model: ', model.summary());

  return model;
}

let modeltest = (model) => {
  /*let uv =  new Float32Array(w*h*inputChannels)
  
  for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
          uv[y*w*2+x*2+0]=y/375
          uv[y*w*2+x*2+1]=x/500
      }
  }
  uv = tf.tensor(uv, [w*h,2])
  let result =model.predict(uv).mul(255).reshape([h,w,3])*/
  let result = tf.tidy(() => model.predict(random))

  visualize(result)
}
  ; (async () => {
    //await save(random.slice([0, 0, 2], [h, w, 3]).mul(255), './temp/rand.jpg')
    //let img = tf.node.decodeJpeg(await fs.promises.readFile('./im.jpg')).slice([0,0,0],[h,w,3]).toFloat().div(255)

    let ys = xys.map(xy => motion.slice([xy, 0], [15, 1]));

    let back = tf.zeros(motion.shape)
    ys.forEach((img, i) => { back = back.maximum(img.pad([[xys[i], (datalen - 15) - xys[i]], [0, 0]])) })

    console.log('back:');
    //visualize(back)
    //await save(back.mul(255), './temp/back.jpg')

    //ys = [].concat(ys.map(y => y.slice([0, 0, 0], [1, 32, 96]))).concat(ys.map(y => y.slice([0, 32, 0], [1, 32, 96]))).concat(ys.map(y => y.slice([0, 64, 0], [1, 32, 96])))

    let model = modelPrep()
    let modelTest = modelPrepTest()
    let modelTest2 = modelPrepTest(true)

    let xs = xys.map(xy => random.slice([xy, 0], [15, inputLen]))
    /*
        mem.push(tf.zeros([16]))
        mem.push(tf.ones([16]))
        mem.push(tf.ones([1]))*/
    //xs = [].concat(xs.map(x => x.slice([0, 0, 0], [1, 32, 96]))).concat(xs.map(x => x.slice([0, 32, 0], [1, 32, 96]))).concat(xs.map(x => x.slice([0, 64, 0], [1, 32, 96])))

    xys.map((xy, i) => {
      let tx = xs[i].unstack()
      let ty = ys[i].unstack()
      tx.forEach((j, h) => { mem.push(tx[h], ty[h]) })
    })
    /*let res = motion.unstack().map(t => {
      return mem.reversePredict(t)
    });*/

    let res = random.unstack().map((t, i, arr) => {
      let res1 = mem.reversePredict(arr[i - 1] || tf.zeros(arr[i].shape))
      let res2 = mem.reversePredict(arr[i])
      let res3 = mem.reversePredict(arr[i + 1] || tf.zeros(arr[i].shape))
      //console.log('res: ', res.outputs);

      let out = new Float32Array(1).fill(1);
      xys.forEach(v => out.fill(0, Math.max(v - i, 0), Math.max(v - i + 15, 0)));

      detect.push(tf.concat([res1.outputs, res2.outputs, res3.outputs]), tf.tensor(out, [1]))
      return res2
    });



    //console.log('res: ', res);
    visualize(tf.concat(res.map(v => v.outputs)))
    xys.map((xy, i) => {
      let tx = xs[i].unstack()
      let ty = ys[i].unstack()
      tx.forEach((j, h) => { mem2.push(tf.concat([tx[h], tf.tensor([0], [1])]), ty[h]) })
    })

    random.unstack().map((t, i, arr) => {

      let res1 = mem.reversePredict(arr[i - 1] || tf.zeros(arr[i].shape))
      let res2 = mem.reversePredict(arr[i])
      let res3 = mem.reversePredict(arr[i + 1] || tf.zeros(arr[i].shape))

      let resd = detect.reversePredict(tf.concat([res1.outputs, res2.outputs, res3.outputs]))
      process.stdout.write('' + (resd.outputs.dataSync()[0] | 0))
      let d = resd.outputs.dataSync()
      xys.forEach(v => d.fill(0, Math.max(v - i, 0), Math.max(v - i + 15, 0)));
      mem2.push(tf.concat([t, tf.tensor(d, [1])]), res2.outputs)

    });

    console.log('');
    let res2 = random.unstack().map((t, i, arr) => {
      let res1 = mem.reversePredict(arr[i - 1] || tf.zeros(arr[i].shape))
      let res2 = mem.reversePredict(arr[i])
      let res3 = mem.reversePredict(arr[i + 1] || tf.zeros(arr[i].shape))

      let resd = detect.reversePredict(tf.concat([res1.outputs, res2.outputs, res3.outputs]))

      let out = mem2.reversePredict(tf.concat([t, tf.tensor([0], [1])]))
      process.stdout.write('' + (resd.outputs.dataSync()[0] | 0))
      return out
    });

    visualize(tf.concat(res2.map(v => v.outputs)))

    /*let res3 = random.unstack().map((t, i, arr) => {
      let res1 = mem2.reversePredict(tf.concat([arr[i - 1] || tf.zeros(arr[i].shape), tf.tensor([0], [1])]))
      let res2 = mem2.reversePredict(tf.concat([arr[i], tf.tensor([0], [1])]))
      let res3 = mem2.reversePredict(tf.concat([arr[i + 1] || tf.zeros(arr[i].shape), tf.tensor([0], [1])]))

      let resd = detect.reversePredict(tf.concat([res1.outputs, res2.outputs, res3.outputs]))
      process.stdout.write('' + (resd.outputs.dataSync()[0] | 0))
      return mem2.reversePredict(tf.concat([t, tf.tensor([0], [1])]))
    });

    visualize(tf.concat(res3.map(v => v.outputs)))*/
    //console.log('mem: ', mem);



    let l0zise = 160
    let lev0 = new associate.Associate(l0zise, 1)
    let lev1 = new associate.Associate(240, l0zise * 4)
    let lev01 = new associate.Associate(l0zise, 1 + lev1.size)

    ys.forEach((y, i) => {
      let ty = y.unstack()
      ty.forEach(j => lev0.push(j))
    })
    ys.map((y, i, arr) => {
      let ty = y.unstack()
      ty.forEach((j, ji, jarr) => {
        if (ji < jarr.length - 4) {
          let l0res = tf.concat([
            lev0.reversePredict(jarr[ji + 0]).mixes,
            lev0.reversePredict(jarr[ji + 1]).mixes,
            lev0.reversePredict(jarr[ji + 2]).mixes,
            lev0.reversePredict(jarr[ji + 3]).mixes
          ])
          lev1.push(l0res)
        }
      })
    })

    ys.map((y, i, arr) => {
      let ty = y.unstack()
      ty.forEach((j, ji, jarr) => {
        if (ji < jarr.length - 4) {
          let l0res = tf.concat([
            lev0.reversePredict(jarr[ji + 0]).mixes,
            lev0.reversePredict(jarr[ji + 1]).mixes,
            lev0.reversePredict(jarr[ji + 2]).mixes,
            lev0.reversePredict(jarr[ji + 3]).mixes
          ])
          let l1mixes = lev1.reversePredict(l0res).mixes
          lev01.push(tf.concat([jarr[ji + 0], l1mixes]))
          lev01.push(tf.concat([jarr[ji + 1], l1mixes]))
          lev01.push(tf.concat([jarr[ji + 2], l1mixes]))
          lev01.push(tf.concat([jarr[ji + 3], l1mixes]))
        }
      })
    })


    let mask = tf.tensor(new Float32Array(l0zise * 4).fill(1).fill(0, l0zise * 3, l0zise * 4))
    let xmask = tf.ones([lev1.size])
    let x1 = lev0.reversePredict(motion.flatten().slice([0], [1])).mixes
    let x2 = lev0.reversePredict(motion.flatten().slice([1], [1])).mixes
    let x3 = lev0.reversePredict(motion.flatten().slice([2], [1])).mixes
    let l1mixes = lev1.reversePredict(tf.concat([x1, x2, x3, tf.zeros(x1.shape)]), mask).mixes

    //let tt = lev01.reversePredict(tf.concat([motion.flatten().slice([0], [1]), l1mixes]), tf.tensor(new Float32Array(21).fill(1).fill(0, 0, 1))).inputs
    //console.log('xx: ', tt, tt.dataSync(), motion.flatten().slice([0], [1]).dataSync());
    let lviz = motion.unstack().map((v, i) => {

      let out = new Float32Array(1).fill(0);
      xys.forEach(v => out.fill(1, Math.max(v - i, 0), Math.max(v - i + 15, 0)));
      let inpmask = tf.tensor(out, [1])
      //console.log('inpmask: ', inpmask.dataSync()[0]);

      let x4t = lev0.reversePredict(v, inpmask).mixes

      l1mixes = lev1.reversePredict(tf.concat([x1, x2, x3, x4t]), tf.concat([tf.ones([x4t.shape[0] * 3]), tf.fill([x4t.shape[0]], inpmask.dataSync()[0])])).mixes

      let x4 = lev01.reversePredict(tf.concat([v, l1mixes]), tf.concat([inpmask, xmask]))

      x1 = x2
      x2 = x3
      x3 = x4.mixes

      return x4.inputs.slice([0], [1])
    })


    //console.log('lviz: ', lviz);
    visualize(tf.concat(lviz))





    return;


    let epochs = 10000;
    let time = new Date()
    for (let e = 0; e < epochs; e++) {
      tf.tidy(() => {
        let lossm = tf.scalar(0)
        for (let i = 0; i < xs.length; i++) {
          let terr = model.optimizer.minimize(() => {
            let error = tf.scalar(0)

            let rIndex = Math.floor(Math.random() * (datalen - 16))
            let uv = random.slice([rIndex, 0], [15, inputLen])
            let inp = model.predict(uv, { batchSize: null })
            let out = new Float32Array(15).fill(1);

            xys.forEach(v => out.fill(-0.01, Math.max(v - rIndex, 0), Math.max(v - rIndex + 15, 0)));
            let restest = modelTest.predict(inp.expandDims(), { batchSize: null })
            let restest2 = modelTest2.predict(inp.expandDims(), { batchSize: null })
            error = error.add(tf.losses.meanSquaredError(restest2.reshape([15, 1]), tf.tensor(out, [15, 1])).mul(0.001))
            error = error.add(restest.relu().mean().mul(0.001))
            let res = model.predict(xs[i], { batchSize: null })
            error = error.add(tf.losses.meanSquaredError(res, ys[i]))

            return error;
          }, true,
            [].concat(...model.layers.map(l => l._trainableWeights.map(v => v.val)))
          )
          let merr = modelTest.optimizer.minimize(() => {
            let error = tf.scalar(0)
            //if (Math.random() < 0.0) {
            let rIndex = Math.floor(Math.random() * (datalen - 16))
            let uv = random.slice([rIndex, 0], [15, inputLen])
            let inp = model.predict(uv, { batchSize: null })
            let out = new Float32Array(15).fill(1);

            xys.forEach(v => out.fill(-0.01, Math.max(v - rIndex, 0), Math.max(v - rIndex + 15, 0)));
            let restest = modelTest.predict(inp.expandDims(), { batchSize: null })
            let restest2 = modelTest2.predict(inp.expandDims(), { batchSize: null })

            error = error.add(tf.losses.meanSquaredError(restest.reshape([15, 1]), tf.tensor(out, [15, 1])))
            error = error.add(tf.losses.meanSquaredError(restest2.reshape([15, 1]), tf.tensor(out, [15, 1])))
            //}
            return error;

          }, true,
            [].concat(...modelTest.layers.map(l => l._trainableWeights.map(v => v.val))).concat(...modelTest2.layers.map(l => l._trainableWeights.map(v => v.val)))
          )
          lossm = lossm.add(merr.div(xs.length).mul(0.001))
          lossm = lossm.add(terr.div(xs.length))
        }

        e % 10 == 9 && console.log(e, lossm.dataSync()[0], new Date() - time)
      })
      e % 10 == 9 && await modeltest(model)
    }

    //await modeltest(model)


  })();


