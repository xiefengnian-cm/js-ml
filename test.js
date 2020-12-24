const math = require('mathjs');
const softmax = function (x) {
  const exp = math.exp(x.map(_ => _ - math.max(x)));
  return exp.map(_ => _ / math.sum(exp));
};

console.log(
  softmax([1, 2, 3, 4])
);

const tanh = function (x) {
  return math.tanh(x);
};

console.log(tanh([1, 2, 3, 4]));

const outer = (matrix_1, matrix_2) => {
  const result = matrix_1.slice();
  return result.map(n => matrix_2.map(m => n * m));
};

console.log(
  outer([1, 2, 3], [4, 5, 6])
);

const d_tanh = function (data) {
  return data.map(n => 1 / math.pow(math.cosh(n), 2));
};

console.log(
  d_tanh([1, 2, 3, 4])
);