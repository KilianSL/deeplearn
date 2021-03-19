using System;
using System.Collections;
using System.Text;

namespace NEA.Tensor
{
    /// <summary>
    /// A class representing a matrix. Uses a single-precision Float32 data type. 
    /// </summary>
    public class Matrix : IEnumerable
    {
        private float[,] data;
        /// <summary>
        /// The dimensions of the matrix, rows x columns
        /// </summary>
        public int[] Shape { get; private set; } // Shape can only be set internally, but the field is publically accessible

        /// <summary>
        /// Initialises an empty matrix with the specified number of rows and columns.
        /// </summary>
        public Matrix(int rows, int columns)
        {
            this.data = new float[rows, columns];
            this.Shape = new int[] { rows, columns };
        }

        /// <summary>
        /// Initialises a matrix with the values in the specified array
        /// </summary>
        /// <param name="data">Array of values to populate the matrix</param>
        public Matrix(float[,] data)
        {
            this.data = data;
            this.Shape = new int[] { data.GetLength(0), data.GetLength(1) };
        }

        /// <summary>
        /// Creates a matrix of the specified dimensions, with all values initialised to 0.
        /// </summary>
        /// <returns>A new matrix initialised with all values at 0.</returns>
        public static Matrix ZeroMatrix(int rows, int columns)
        {
            return new Matrix(rows, columns); // float arrays initialise to 0.0f by default, function simply provided for user ease
        }

        /// <summary>
        /// Creates a matrix of the specified dimensions populated with values drawn from a random Gaussian distribution.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix.</param>
        /// <param name="columns">The number of columns in the matrix.</param>
        /// <param name="mean">The mean of the distribution. Default 0</param>
        /// <param name="stdDev">The standard deviation of the distribution. Default 1</param>
        /// <returns>A new matrix intialised with the specified parameters.</returns>
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
        /// <summary>
        /// Gets a single value from the first column of the matrix. Only works with column matricies.
        /// </summary>
        /// <param name="idx">The row index of the required item.</param>
        /// <returns>The float at [idx,0].</returns>
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

        /// <summary>
        /// Specifies a single value at the desired row and column indices. Allows get and set operations. 
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <param name="col">The column index.</param>
        /// <returns>The float at [row,col].</returns>
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
        // Enumerates through each column then cycles to the next row
        /// <summary>
        /// Gets an enumerable collection from this matrix.
        /// </summary>
        /// <returns>An IEnumerator representing the matrix.</returns>
        public IEnumerator GetEnumerator()
        {
            foreach (var element in data)
            {
                yield return element; // returns a value while retaining function scope to resume execution and return the next value
            }
        }

        // IsEqual - Returns true if the argument matrix is equal to this matrix
        // Works by comparing the matrix string representations
        /// <summary>
        /// Checks two objects are equal.
        /// </summary>
        /// <param name="obj">The object to compare this matrix to.</param>
        /// <returns>A boolean equality value.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Matrix  )
            {
                return this.ToString() == obj.ToString();
            }
            else
            {
                return false;
            }
        }

        // Override of GetHashCode needed for proper performance of Assert.IsEqual in unit testing - hash code is used to check equality
        // Implemented as the first 32-bit integer of the MD5 hash of the matrix string representation.
        /// <summary>
        /// Gets the hash value for this matrix.
        /// </summary>
        /// <returns>The hash value of this matrix.</returns>
        public override int GetHashCode()
        {
            string matrixIdentifier = this.ToString();
            using var md5 = System.Security.Cryptography.MD5.Create();
            // The utf8 encoding of the string as a byte array
            byte[] inputBytes = Encoding.UTF8.GetBytes(matrixIdentifier);
            byte[] hashBytes = md5.ComputeHash(inputBytes);
            // MD5 implements the IDisposable interface, so needs to be explicitely disposed to free up allocated resources
            md5.Dispose();

            // First 4 bytes (32bits) converted to a single Int32 hash code
            // Array is treated in a big-endian order
            int hashCode = 0;
            for (int i = 3; i >= 0; i--)
            {
                hashCode += hashBytes[3 - i];
                hashCode <<= 8 * i; 
            }
            
            return hashCode;
            
        }

        // Check matrix shape is equal to this matrix, throws error on false (DRY principal for all functions where matrix dimensions need to be equal)
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

        /// <summary>
        /// Adds matrix A to this matrix.
        /// </summary>
        public void Add(Matrix A)
        {
            checkShapeEqual(A);
            var newData = new float[Shape[0], Shape[1]];
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    newData[i, j] = this[i, j] + A[i, j];
                }
            }
            data = newData;
        }

        /// <summary>
        /// Performs the Hadamard (elementwise) product between this matrix and matrix A.
        /// </summary>
        public void Hadamard(Matrix A)
        {
            checkShapeEqual(A);
            var newData = new float[Shape[0], Shape[1]];
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    newData[i, j] = this[i, j] * A[i, j];
                }
            }
            data = newData;
        }

        /// <summary>
        /// Returns the dot product of this matrix and matrix A.
        /// </summary>
        /// <returns>The dot product.</returns>
        public float Dot(Matrix A)
        {
            checkShapeEqual(A);
            float result = 0;
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    result += this[i, j] * A[i, j];
                }
            }
            return result;
        }

        /// <summary>
        /// Transforms this matrix by matrix A.
        /// </summary>
        public void Transform(Matrix A)
        {
            if (Shape[0] == A.Shape[1])
            {
                var result = new float[A.Shape[0], Shape[1]];
                for (int i = 0; i < A.Shape[0]; i++)
                {
                    for (int j = 0; j < Shape[1]; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < A.Shape[1]; k++)
                        {
                            sum += A[i, k] * this[k, j];
                        }
                        result[i, j] = sum;
                    }
                }
                data = result;
                Shape[0] = A.Shape[0];
            }
            else
            {
                throw new Exception("Incorrect dimensions for matrix transform");
            }
        }

        // Flatten and reshape method - reshapes matrix values to the specified shape
        // Uses for loops to "flatten" the matrix, then reads to the appropriate row and column of the target matrix
        /// <summary>
        /// Reshapes the matrix to fit the specified dimensions. 
        /// </summary>
        public void Reshape(int rows, int cols)
        { 
            if (Shape[0] * Shape[1] != rows * cols)
            {
                throw new Exception("Matrix does not fit reshape dimensions");
            }
            var flattenedData = new float[rows * cols];
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    flattenedData[i * j] = this[i, j];
                }
            }
            var newData = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    newData[i, j] = flattenedData[i * j];
                }
            }
            data = newData;
            Shape = new int[] { rows, cols };
        }

        // Matrix transpose algorithm
        /// <summary>
        /// Transposes the matrix.
        /// </summary>
        public void Transpose()
        {
            var newData = new float[Shape[1], Shape[0]];
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    newData[j, i] = this[i, j];
                }
            }
            data = newData;
            Shape = new int[] { Shape[1], Shape[0] };
        }

        /// <summary>
        /// Returns a new float array storing the data from the matrix.
        /// </summary>
        /// <returns>A new float array storing the data from the matrix.</returns>
        public float[,] ToArray()
        {
            return data;
        }

        /// <summary>
        /// Generates a string representation of this matrix.
        /// </summary>
        /// <returns>A string representing this matrix.</returns>
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