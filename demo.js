const math = require('mathjs');
const path = require('path');
const fs = require('fs');
const { Matrix } = require('ml-matrix');

/**
 * 
 * @param {Matrix} vector_1 
 * @param {Matrix} vector_2 
 */
Matrix.outer = function (vector_1, vector_2) {
  const { rows : r1, columns : c1 } = vector_1;
  const { rows : r2, columns : c2 } = vector_2;
  const result = new Matrix(c1*r1,c2*r2);
  let v2_arr = [];
  for(let i = 0; i < vector_2.rows;i++){
    v2_arr.push(
      ...vector_2.getRow(i)
    )
  }
  const v2_vector = new Matrix([v2_arr]);

  let result_row = 0;
  for (let i = 0;i < vector_1.rows;i++) {
    for(let j = 0;j< vector_1.columns;j++){
      const n = vector_1.get(i,j);
      result.setRow(result_row,Matrix.mul(v2_vector,n));
      result_row++;
    }
  }
  return result;
};

// 数据集
const dataset_path = './MNIST';
const train_img_path = path.resolve(__dirname, dataset_path, 'train-images-idx3-ubyte');
const train_lab_path = path.resolve(__dirname, dataset_path, 'train-labels-idx1-ubyte');
const test_img_path = path.resolve(__dirname, dataset_path, 't10k-images-idx3-ubyte');
const test_lab_path = path.resolve(__dirname, dataset_path, 't10k-labels-idx1-ubyte');

const train_num = 50000;
const valid_num = 10000;
const test_num = 10000;

// const tmp_img = math.reshape(, [28, 28]);

const getImgs = (path) => {
  const file = fs.readFileSync(path).slice(16);
  const uint8array = new Uint8Array(file);
  const array = [];
  let i = 0;
  while (i < uint8array.length) {
    array.push(uint8array[i]);
    i++;
  }
  const uint8 = array;
  const count = uint8.length / (28 * 28);
  const imgs = [];
  for (let i = 0; i < count; i++) {
    const b_index = i * 28 * 28;
    const e_index = (i + 1) * 28 * 28;
    const img = uint8.slice(b_index, e_index);
    imgs.push(new Matrix([img]));
  }
  return imgs;
};
const getLabel = (path) => {
  const file = fs.readFileSync(path).slice(8);
  const uint8array = new Uint8Array(file);
  const array = [];
  let i = 0;
  while (i < uint8array.length) {
    array.push(uint8array[i]);
    i++;
  }
  return array;
};

const t_imgs = getImgs(train_img_path);
const train_img = t_imgs.slice(0, train_num);
const valid_img = t_imgs.slice(-valid_num);
const test_img = getImgs(test_img_path);

const t_labs = getLabel(train_lab_path);
const train_lab = t_labs.slice(0, train_num);
const valid_lab = t_labs.slice(-valid_num);
const test_lab = getLabel(test_lab_path);
/**
 * 
 * @param {Matrix} x 
 */
const tanh = function (x) {
  return Matrix.tanh(x);
};

/**
 * 
 * @param {Matrix} x 
 */
const softmax = function (x) {
  const exp = Matrix.exp(
    Matrix.sub(x, new Matrix(x.rows, x.columns).fill(x.max()))
  );
  return Matrix.div(exp, exp.sum());
};


const dimension = [28 * 28, 10];
const activation = [tanh, softmax];
const distribution = [
  {
    'b': [0, 0]
  },
  {
    'b': [0, 0],
    'w': [
      -math.sqrt(6 / (dimension[0] + dimension[1])),
      math.sqrt(6 / (dimension[0] + dimension[1]))
    ]
  }
];

const init_parameters_b = function (layer) {
  const dist = distribution[layer]['b'];
  return Matrix.rand(1, dimension[layer], {
    random: () => math.random() * (dist[1] - dist[0]) + dist[0]
  });
};
const init_parameters_w = function (layer) {
  const dist = distribution[layer]['w'];
  return Matrix.rand(dimension[layer - 1], dimension[layer], {
    random: () => math.random() * (dist[1] - dist[0]) + dist[0],
  });
};

const init_parameters = function () {
  const parameter = [];
  for (let i = 0; i < distribution.length; i++) {
    const layer_parameter = {};
    for (const key in distribution[i]) {
      if (key === 'b') {
        layer_parameter['b'] = init_parameters_b(i);
        continue;
      }
      if (key === 'w') {
        layer_parameter['w'] = init_parameters_w(i);
        continue;
      }
    }
    parameter.push(layer_parameter);
  }
  return parameter;
};
/**
 * 
 * @param {Matrix} img 
 * @param {any} init_parameters 
 */
const predict = function (img, init_parameters) {
  const l0_in = Matrix.add(img, parameters[0]['b']);
  const l0_out = activation[0](l0_in);
  const l1_in = Matrix.add(parameters[1]['b'], l0_out.mmul(parameters[1]['w']));
  const l1_out = activation[1](l1_in);
  return l1_out;
};
// 训练
/**
 * 
 * @param {Matrix} data 
 */
const d_softmax = function (data) {
  const sm = softmax(data);
  return Matrix.sub(Matrix.diag(sm.getRow(0)), Matrix.outer(sm, sm));
};

/**
 * 
 * @param {Matrix} data 
 */
const d_tanh = function (data) {
  return Matrix.ones(data.rows, data.columns).div(Matrix.cosh(data).pow(2));
};

const differential = {
  softmax: d_softmax,
  tanh: d_tanh,
};

const onehot = Matrix.identity(dimension.slice(-1)[0]);

/**
 * 
 * @param {Matrix} img 
 * @param {number} lab 
 * @param {any} parameters 
 */
const sqr_loss = function (img, lab, parameters) {
  const y_pred = predict(img, parameters);
  const y = onehot.getRowVector(lab);
  const diff = Matrix.sub(y, y_pred);
  return diff.dot(diff);
};

/**
 * 
 * @param {Matrix} img 
 * @param {number} lab 
 * @param {any} init_parameters 
 */
const grad_parameters = function (img, lab, init_parameters) {
  const l0_in = Matrix.add(img, parameters[0]['b']);      // 1 * 784
  const l0_out = activation[0](l0_in);        // 1 * 784
  const l1_in = Matrix.add(parameters[1]['b'], l0_out.mmul(parameters[1]['w'])); // 1 * 10
  const l1_out = activation[1](l1_in);        // 1 * 10
  // w : 784 * 10, b1 : 1 * 10, b0 : 1 * 784
  const diff = Matrix.sub(onehot.getRowVector(lab), l1_out);  // 1 * 10
  // 1*10 dot 10 * 10   => 1 *10
  const act_1 =diff.mmul(differential[activation[1].name](l1_in)); //应为1*10
  const grad_b1 = Matrix.mul(act_1, -2);
  const grad_w1 = Matrix.mul(Matrix.outer(l0_out, act_1), -2);
  // w :784 * 10 dot 1 * 10 error  => 1*784
  // 1 * 784 <= 1*784 * 1*784
  const grad_b0 = Matrix.mul(differential[activation[0].name](l0_in),act_1.mmul(parameters[1]['w'].transpose())).mul(-2);
  
  return {
    'w1': grad_w1,
    'b1': grad_b1,
    'b0': grad_b0,
  };
};

let h = 0.001;
// for (let i in Array(10).fill(0)) {
//   const img_i = math.randomInt(0, train_num);
//   const test_parameters = init_parameters();
//   const derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)['b1'];
//   const value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters);
//   const newVec = test_parameters[1]['b'].getColumnVector(i).add(h);
//   test_parameters[1]['b'].setColumn(i,newVec);
//   const value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters);
//   console.log(Matrix.sub(derivative.getColumnVector(i), (value2 - value1)/h).data[0]);
// }
// for (let i in Array(10).fill(0)) {
//   const img_i = math.randomInt(0, train_num);
//   const test_parameters = init_parameters();
//   const derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)['b0'];
//   const value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters);
//   const newVec = test_parameters[0]['b'].getColumnVector(i).add(h);
//   test_parameters[0]['b'].setColumn(i,newVec);
//   const value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters);
//   console.log(Matrix.sub(derivative.getColumnVector(i), (value2 - value1)/h).data[0]);
// }
const valid_loss = function (parameters) {
  let loss_accu = 0;
  for (const img_i in new Array(valid_num).fill(0)) {
    loss_accu += sqr_loss(valid_img[img_i], valid_lab[img_i], parameters);
  }
  return loss_accu;
};


const valid_accuracy = function (parameters) {
  const correct = [];
  for (let img_i in Array(valid_num).fill(0)) {
    correct.push(
      predict(valid_img[img_i], parameters).maxIndex()[1] == valid_lab[img_i]
    );
  }
  console.log(`validation accuracy : ${correct.filter(_=>_).length / correct.length}`)
};

const batch_size = 100;

const train_batch = function(current_batch,parameters){
  const grad_accu = grad_parameters(
    train_img[current_batch * batch_size + 0],
    train_lab[current_batch * batch_size + 0],
    parameters
  )
  for(let img_i = 1; img_i < batch_size;img_i++){
    const grad_tmp = grad_parameters(
      train_img[current_batch * batch_size + img_i],
      train_lab[current_batch * batch_size + img_i],
      parameters
    )
    for(const key in grad_tmp){
      grad_accu[key].add(grad_tmp[key]);
    }
  }
  for(const key in grad_accu){
    grad_accu[key].div(batch_size);
  }
  return grad_accu;
}

const combine_parameters = function(parameters,grad,learn_rate){
    const parameter_tmp = JSON.parse(JSON.stringify(parameters));
    parameter_tmp[0]['b'] = new Matrix(parameter_tmp[0]['b']).sub(grad['b0'].mul(learn_rate))
    parameter_tmp[1]['b'] = new Matrix(parameter_tmp[1]['b']).sub(grad['b1'].mul(learn_rate))
    parameter_tmp[1]['w'] = new Matrix(parameter_tmp[1]['w']).sub(grad['w1'].mul(learn_rate))
    return parameter_tmp
}

const learn_self = function(learn_rate){
  for(let i = 0 ; i < parseInt(train_num/batch_size);i++){
    if(i%100 == 99){
      console.log(`running batch ${i + 1}/${parseInt(train_num/batch_size)}`);
    }
    grad_tmp = train_batch(i, parameters)
    parameters = combine_parameters(parameters, grad_tmp, learn_rate)
  }
}

const run_train = function(){
  const learn_rate = 1
  const model = JSON.parse(fs.readFileSync('./model',{encoding:'utf-8'}));
        model[0]['b'] = new Matrix(model[0]['b'])
        model[1]['b'] = new Matrix(model[1]['b'])
        model[1]['w'] = new Matrix(model[1]['w'])
  let parameters = model;
  valid_accuracy(parameters)
  learn_self(learn_rate)
  valid_accuracy(parameters)
  fs.writeFileSync('./model_new',JSON.stringify(parameters),{encoding:'utf-8'})
}


/**
 * 
 * @param {number[]} img 
 * @param {any} parameters 
 */
const exports_predict = function (img, parameters) {
  img = new Matrix([img])
  const l0_in = Matrix.add(img, parameters[0]['b']);
  const l0_out = activation[0](l0_in);
  const l1_in = Matrix.add(parameters[1]['b'], l0_out.mmul(parameters[1]['w']));
  const l1_out = activation[1](l1_in);
  return l1_out;
};

module.exports={
  get_random_train_img : ()=>{
    const r = math.randomInt(0,train_num);
    return {
      img : train_img[r],
      lab : train_lab[r],
    }
  },
  get_random_test_img : ()=>{
    const r = math.randomInt(0,test_num);
    return {
      img : test_img[r],
    }
  },
  predict: exports_predict,
  load_model : (path)=>{
    try{
      const model = JSON.parse(fs.readFileSync(path,{encoding:'utf-8'}));
      model[0]['b'] = new Matrix(model[0]['b'])
      model[1]['b'] = new Matrix(model[1]['b'])
      model[1]['w'] = new Matrix(model[1]['w'])
      return model;
    } catch(err){
      console.log(err)
    } 
  }
}