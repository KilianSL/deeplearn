using System;
using System.Text;
using System.Collections;

namespace NEA.Tensor
{
    public class Matrix : IEnumerable, IEquatable<Matrix>
    {
        public float[,] data;
        public int[] Shape { get; private set; }

        // Constructor for empty matrix with specified shape
        public Matrix(int rows, int columns)
        {
            this.data = new float[rows, columns];
            this.Shape = new int[] { rows, columns };
        }

        // Constructor for matrix object prefilled with values from a float array
        public Matrix(float[,] data)
        {
            this.data = data;
            this.Shape = new int[] { data.GetLength(0), data.GetLength(1) };
        }

        // Higher-order constructor that returns a matrix filled with 0s
        public static Matrix ZeroMatrix(int rows, int columns)
        {
            return new Matrix(rows, columns); // float arrays initialise to 0.0f by default, function simply provided for user ease
        }

        // Higher-order constructor that returns matrix filled with numbers drawn from a guassian distrbution
        public static Matrix GaussianMatrix(int rows, int columns, float mean = 0, float stdDev = 1)
        {
            float[,] data = new float[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i, j] = MatrixUtils.RandomGaussian(mean, stdDev)[0];
                }
            }
            return new Matrix(data);
        }

        // Indexing methods - allow the matrix to be indexed like an array
        // single index = will take data from that row (assuming it is a column matrix)
        // double index = will index that location in the matrix in row-major order
        // will allow both GET and SET operations
        public float this[int idx]
        {
            get
            {
                if (Shape[1] == 1) // if there is only 1 column dimension
                {
                    return data[idx, 0];
                }
                else
                {
                    throw new Exception("Matrix has more than one column - unsuitable indexing");
                }
            }
            set
            {
                if (Shape[1] == 1) // if there is only 1 column dimension
                {
                    data[idx, 0] = value;
                }
                else
                {
                    throw new Exception("Matrix has more than one column - unsuitable indexing");
                }
            }
        }

        public float this[int row, int col] // Error handling for this will be handled by array implementation (IndexOutOfRange Error)
        {
            get
            {
                return data[row, col];
            }
            set
            {
                data[row, col] = value;
            }
        }

        // IEnumerable - Get Enumerator - Returns enumerable collection meaning the Matrix class can be used in foreach loops etc.
        public IEnumerator GetEnumerator()
        {
            foreach (var element in data)
            {
                yield return element; // returns a value while retaining function scope to resume execution and return the next value
            }  
        }

        // IEquatable - IsEqual - Returns true if the argument matrix is equal to this matrix
        public bool Equals(Matrix A)
        {
            if (Shape[0] == A.Shape[0] && Shape[1] == A.Shape[1])
            {
                for (int i = 0; i < Shape[0]; i++)
                {
                    for (int j = 0; j < Shape[1]; j++)
                    {
                        if (this[i,j] != A[i,j])
                        {
                            return false;
                        }
                    }
                }
                return true; 
            }
            return false;
        }

        // Check matrix shape is equal to this matrix, throws error on false (DRY principal for all functions where matrix dimensions need to be equal
        private void checkShapeEqual(Matrix A)
        {
            if (Shape[0] == A.Shape[0] && Shape[1] == A.Shape[1])
            {
                return;
            }
            else
            {
                throw new Exception("Matrix dimensions do not comform");
            }
        }

        // Addition Method - adds this instance of a matrix to another matrix
        public void Add(Matrix A)
        {
            checkShapeEqual(A);
            var newData = new float[Shape[0], Shape[1]];
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    newData[i, j] = data[i, j] + A[i, j];
                }
            }
            data = newData;
        }

       // Elementwise product
        public void Hadamard(Matrix A)
        {
            checkShapeEqual(A);
            var newData = new float[Shape[0], Shape[1]];
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    newData[i, j] = data[i, j] * A[i, j];
                }
            }
            data = newData;
        }
        
        // Dot product method - returns a float dot product
        public float Dot(Matrix A)
        {
            checkShapeEqual(A);
            float result = 0;
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    result = +data[i, j] * A[i, j];
                }
            }
            return result;
        }

        // Flatten and reshape method - reshapes matrix values to the specified shape
        public void Reshape(int rows, int cols)
        {
            if (Shape[0]*Shape[1] != rows * cols)
            {
                throw new Exception("Matrix does not fit reshape dimensions");
            }
            var newData = new float[rows, cols];
            int currentRow = 0;
            int currentCol = 0;
            for (int i = 0; i < rows * cols; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    newData[currentRow, currentCol] = data[i, j];
                    currentCol++;
                    if (currentCol==cols)
                    {
                        currentRow++;
                        currentCol = 0;
                    }
                }
            }
            data = newData;
        }

        public override string ToString()
        {
            var sb = new StringBuilder("[");

            for (int i = 0; i < Shape[0]; i++)
            {
                sb.Append("[");
                for (int j = 0; j < Shape[1]; j++)
                {
                    sb.Append(data[i, j]);
                    if (j != Shape[1] - 1)
                    {
                        sb.Append(", \t");
                    }
                }
                sb.Append("]");
                if (i != Shape[0] - 1)
                {
                    sb.Append(','); ;
                    sb.AppendLine();
                }
            }
            sb.Append("]");
            return sb.ToString();
        }
    }
}