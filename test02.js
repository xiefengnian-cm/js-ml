const { Matrix } = require('ml-matrix');

const A = new Matrix([[1, 2, 3, 4]]);
const B = new Matrix([[4, 5, 6, 7]]);

console.log(Matrix.add(A, B).data);

console.log(
  Matrix.identity(10,10)
)
new Matrix([[1,2,3]]).sub(new Matrix([[1,2,3]]))



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
console.log(
  Matrix.outer(new Matrix([[1,2,3,4]]),new Matrix([[1],[2],[3]]))
)
console.log(
  Matrix.outer(
    new Matrix([[1,2,3,4],[2,2,2,2]]),new Matrix([[1,1],[2,2],[3,3]])
  )
)
debugger;