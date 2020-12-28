const path = require('path');
const express = require('express');
const body_parser = require('body-parser');
const json_parser = body_parser.json()
const { get_random_train_img, load_model, predict, get_random_test_img } = require('./demo');
const app = express();
app.listen(3900);

app.use(express.static('./public'));


const model = load_model(path.resolve(__dirname,'./model_new_255'));


app.get('/get_random_train_img',(req,res)=>{
  res.send(
    get_random_train_img()
  )
})

app.post('/predict',json_parser,(req,res)=>{
  const img = req.body.img;
  if(img.length !== 784){
    return res.send('error');
  }
  res.send(predict(img,model).maxIndex()[1].toString())
})

app.get('/get_random_test_img',(req,res)=>{
  res.send(
    get_random_test_img()
  )
})