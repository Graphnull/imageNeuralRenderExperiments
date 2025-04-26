let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');


console.log(' files:', fs.readdirSync('./nebulas/'));
tf.enableProdMode()
let inputChannels = 24 + 21
//let random = tf.rand([375,500,inputChannels], Math.random)
function mulberry32(a) {
    return function() {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}
let rand = mulberry32(0)
let w = 512;
let h = 512;
/**
 * @type {tf.Tensor}
 */
let uvdata = new Float32Array(w * h * 2)
for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
        uvdata[y * w * 2 + x * 2 + 0] = y / (h/2)-0.5
        uvdata[y * w * 2 + x * 2 + 1] = x / (w/2)-0.5
    }
}
let dataset = tf.tidy(()=>fs.readdirSync('./nebulas/').map(f=> 
    tf.node.decodeJpeg(fs.readFileSync('./nebulas/'+f)).resizeBilinear([w,h]).toFloat().div(255)
))
let fuv = tf.tensor(uvdata, [h, w, 2])
uv = fuv.slice([0, 0], [96, 96])

let fuv1 = fuv.slice([0, 0, 0], [h, w, 1])
let fuv2 = fuv.slice([0, 0, 1], [h, w, 1])
let random = tf.concat([
    fuv.sub(0.5).abs(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(5200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(5200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(1600).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(1600).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(1200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(100).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(100).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(100).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(50).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(50).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(50).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(50).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(25).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(25).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(25).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(12).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(12).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(12).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3).sin(),
], -1)
random= tf.tensor(random.dataSync().map(v=>v+Math.random()*0.00),[h,w,inputChannels])
let save = async (tensor, name) => {
    let bin = await tf.node.encodeJpeg(tensor.minimum(255).maximum(0), "rgb", 100)
    await fs.promises.writeFile(name, bin);
}


let modeltest = async (model, modelEmbed,k) => {

    let ind = new Float32Array(dataset.length)
    ind[k]=1//k/10
    //ind[5]=1-k/10
    //ind[0]=1
    //let onehot = tf.randomUniform([1,1,1, dataset.length])
    let onehot = tf.tensor(ind,[1,1,1,dataset.length])
    //let onehot = tf.tensor([0.95688920,0.3020564,0.59081429,0.68059,0.12054583,0.17261013,0.3246,
    //    0.37351691,0.0966565,0.00490524,0.544575,0.6052898,0.003812783,0.243843,0.759258,0.9662610
    //],[1,1,1,dataset.length])
    console.log('onehot :', onehot.dataSync());
    let embv = modelEmbed.predict(onehot)
    let inp = tf.concat(new Array(w*h).fill(0).map(v=> embv)).reshape([1,h,w,dataset.length])

    let result = tf.tidy(() => model.predict(tf.concat([random.expandDims(), inp],-1)).mul(255).reshape([h, w, 3]))
    return await save(result, './temp/result'+k+'.jpg')
}

    ; (async () => {
    
        let model = await tf.loadLayersModel('file://G:\\Мой диск\\nebulas\\model\\model.json')
        let modelEmbed = await tf.loadLayersModel('file://G:\\Мой диск\\nebulas\\modelEmbed\\model.json')

        for(let i=0;i<16;i++){
            await modeltest(model,modelEmbed, i)
        }
    



    })();