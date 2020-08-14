using System;
using System.Collections.Generic;
using System.Text;

namespace NEA.Tensor
{
    static class MatrixUtils
    {
        public static float[] RandomGaussian(float mean, float stdDev) // Returns an array of 2 random gaussian variables generated using the Marsaglia-Polar method
        {
            Random random = new Random();
            float randomFloat() { return (float)random.NextDouble(); } // gets a random float value from U(0,1)
            float s = 5;
            float u1 = 1;
            float u2 = 1;
            while (s > 1)
            {
                u1 = randomFloat();
                u2 = randomFloat();
                u1 = 2 * u1 - 1;
                u2 = 2 * u2 - 2;
                s = MathF.Pow(u1, 2) + MathF.Pow(u2, 2);
            }
            u1 = MathF.Sqrt((-2 * MathF.Log(s)) / s) * u1;
            u2 = MathF.Sqrt((-2 * MathF.Log(s)) / s) * u2;
            return new float[] { u1 * stdDev + mean, u2 * stdDev + mean };
        }
    }
    public class Matrix
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
        public static Matrix GaussianMatrix(int rows, int columns, float mean=0, float stdDev = 1)
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
                if (Shape[1]==1) // if there is only 1 column dimension
                {
                    return data[idx, 0];
                } else
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

        // Addition Method - adds this instance of a matrix to another matrix
        public void Add(Matrix A)
        {
            if (Shape[0] == A.Shape[0] && Shape[1] == A.Shape[1])
            {
                var newData = new float[Shape[0], Shape[1]];
                for (int i = 0; i < Shape[0]; i++)
                {
                    for (int j = 0; j < Shape[1]; j++)
                    {
                        newData[i, j] = data[i, j] + A[i, j];
                    }
                }
                data = newData;
            } else
            {
                throw new Exception("Matrix dimensions do not comform");
            }
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
