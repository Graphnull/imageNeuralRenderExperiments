let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');
tf.enableProdMode ()

let inputChannels = 24+21
//let random = tf.rand([375,500,inputChannels], Math.random)


/**
 * @type {tf.Tensor}
 */
let uvdata =  new Float32Array(500*375*2)
for(let y=0;y<375;y++){
    for(let x=0;x<500;x++){
        uvdata[y*500*2+x*2+0]=y/375
        uvdata[y*500*2+x*2+1]=x/500
    }
}
let fuv = tf.tensor(uvdata, [375,500,2])
uv = fuv.slice([0,0],[100,100])
let random = tf.concat([
    fuv,
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(5200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(6200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(5200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(6200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(3200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(3200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(1600).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(1600).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(1200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(800).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(800).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(800).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(800).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(800).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(800).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(400).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(400).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(400).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(400).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(400).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(400).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(200).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(100).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(100).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(100).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(50).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(50).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(50).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(50).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(25).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(25).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(25).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(12).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(12).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(12).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(6).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(6).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(6).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(3).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(3).sin(), 
    fuv.slice([0,0,0],[375,500,1]).mul(Math.random()).add(fuv.slice([0,0,1],[375,500,1]).mul(Math.random()*2-1)).mul(3).sin(), 
],-1)

let save = async (tensor, name)=>{
    let bin = await tf.node.encodeJpeg(tensor.minimum(255).maximum(0), "rgb", 100)
    await fs.promises.writeFile(name, bin);
}
let xys = [[0,0],[120,120],[275,400],[230,277],[150,320],[140,300],[120,370], [40,200],[50,350],[45,270],[150,50],[230,70]];

let modelPrep = ()=>{
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 190, activation:'linear',inputShape: [inputChannels]}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
    model.add(tf.layers.dense({units: 120, activation:'linear',}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
    model.add(tf.layers.dense({units: 80, activation:'linear',}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
    //model.add(tf.layers.batchNormalization({}));
    model.add(tf.layers.dense({units: 50, activation:'linear',}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
    model.add(tf.layers.dense({units: 30, activation:'linear',}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
    model.add(tf.layers.dense({units: 20, activation:'linear',}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
    model.add(tf.layers.dense({units: 12, activation:'linear',}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
    //model.add(tf.layers.batchNormalization({}));

    model.add(tf.layers.dense({units: 6, activation:'linear',}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
    //model.add(tf.layers.batchNormalization({}));
    model.add(tf.layers.dense({units: 3, activation:'linear',}));
    model.add(tf.layers.leakyReLU({alpha:0.01}));
  
    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'adam'});

    return model;
}
let modeltest = (model)=>{
    let w =500
    let h =375;
    /*let uv =  new Float32Array(w*h*inputChannels)
    
    for(let y=0;y<h;y++){
        for(let x=0;x<w;x++){
            uv[y*w*2+x*2+0]=y/375
            uv[y*w*2+x*2+1]=x/500
        }
    }
    uv = tf.tensor(uv, [w*h,2])
    let result =model.predict(uv).mul(255).reshape([h,w,3])*/
    let result =tf.tidy(()=>model.predict(random.reshape([h*w, inputChannels])).mul(255).reshape([h,w,3]))
    
    return save(result, './temp/result.jpg')
}

;(async ()=>{
    await save(random.slice([0,0,2],[375,500,3]).mul(255), './temp/rand.jpg')
    let img = tf.node.decodeJpeg(await fs.promises.readFile('./im.jpg')).toFloat()
    

    let dataset = xys.map(xy=>img.slice(xy,[100,100]));
    
    let model = modelPrep()

    //const xs = tf.stack(xys.map(xy=> uv.add([xy[0]/375, xy[1]/500]) )).reshape([12*100*100,2]);
    let xs = tf.stack( xys.map(xy=>random.slice(xy,[100,100])) ).reshape([12*100*100,inputChannels])

    const ys = tf.stack(dataset).reshape([12*100*100,3]).mul(1/255);
    let timeout;
    let testTemp = () => {
        timeout = setTimeout(() => {
        modeltest(model).then(testTemp)
        },10000)
       }
    testTemp();
    await model.fit(xs, ys,{
        batchSize: 100*50,
        epochs: 1000,
        verbose:0,
        callbacks:{
            onEpochEnd:(epoch, logs)=>{
                console.log(epoch, logs.loss)
            }
        }
    })
    await modeltest(model)
    clearTimeout(timeout)

    dataset.forEach((img,i)=>save(img, './temp/'+i+'.jpg'))

    let back = tf.zeros(img.shape)

    dataset.forEach((img,i)=>{back = back.maximum(img.pad( [[xys[i][0], 275-xys[i][0]], [xys[i][1], 400-xys[i][1]], [0,0]] ))})
    await save(back, './temp/back.jpg')
})();