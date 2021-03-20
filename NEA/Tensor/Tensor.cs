using System;
using System.Linq;
using System.Text;

namespace NEA.Tensor
{
    public class Tensor
    {
        private Matrix[] data; // The data stored in the tensor
        public int[] Shape { get; private set; } // The shape of the tensor in order [batch, matrix_rows, matrix_columns]

        public Tensor(int batches, int rows, int columns)
        {
            var data = new Matrix[batches];
            for (int i = 0; i < batches; i++)
            {
                data[i] = new Matrix(rows, columns);
            }
            this.data = data;
            this.Shape = new int[] { batches, rows, columns };
        }

        // Indexing methods for the Tensor, implemented as per the documentation
        /// <summary>
        /// Gets the batch at the specified index.
        /// </summary>
        /// <param name="idx">The index in the tensor.</param>
        /// <returns>A matrix representing the batch.</returns>
        public Matrix this[int idx]
        {
            get
            {
                return data[idx];
            }
            set
            {
                data[idx] = value; // will handle error in the case where the matrix is not an Nx1 vector
            }
        }

        public float this[int row, int col]
        {
            get
            {
                if (Shape[0] == 1)
                {
                    return data[0][row, col]; // will handle error in the case where the matrix is not an Nx1 vector
                }
                else
                {
                    throw new Exception("Unsuitable Tensor Indexing");
                }
            }
            set
            {
                if (Shape[0] == 1)
                {
                    data[0][row, col] = value;
                }
                else
                {
                    throw new Exception("Unsuitable Tensor Indexing");
                }
            }
        }

        public float this[int batch, int row, int col]
        {
            get
            {
                return data[batch][row, col];
            }
            set
            {
                data[batch][row, col] = value;
            }
        }

        // Gets a specific item from the batched tensor
        public Matrix GetItem(int idx)
        {
            return data[idx];
        }

        // Sets a specific batch to an instance of a matrix
        public void SetItem(int idx, Matrix m)
        {
            data[idx] = m;
        }

        public override int GetHashCode()
        {
            string tensorIdentifier = "";
            foreach (var matrix in data)
            {
                tensorIdentifier += matrix.ToString();
            }
            using var md5 = System.Security.Cryptography.MD5.Create();
            // The utf8 encoding of the string as a byte array
            byte[] inputBytes = Encoding.UTF8.GetBytes(tensorIdentifier);
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

        public override bool Equals(object obj)
        {
            if (obj is Tensor)
            {
                return this.ToString() == obj.ToString();
            }
            else
            {
                return false;
            }
        }

        // Concatenates the string representations of all the matricies in the tensor, separated by line breaks and comma
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder("[");
            foreach (var matrix in data)
            {
                sb.Append(matrix.ToString());
                sb.Append(",\n");
            }
            sb.Append(']');
            return sb.ToString();
        }

        // Tensorised Linear Algebra methods

        // Checks that two tensors are of equal dimensions.
        // This method is only called in situations where mismatched tensors would cause a fatal error, hence it throws an exception
        private void checkShapeEqual(Tensor A)
        {
            if (Enumerable.SequenceEqual(Shape, A.Shape))
            {
                return;
            }
            else
            {
                throw new Exception("Tensor dimensions do not conform");
            }
        }

        // Checks that this tensor and a matrix are of equal dimensions.
        // This method is only called in situations where mismatched dimensions would cause a fatal error, hence it throws an exception
        private void checkShapeEqual(Matrix A)
        {
            if (A.Shape[0] == Shape[1] && A.Shape[1] == Shape[2])
            {
                return;
            }
            else
            {
                throw new Exception("Tensor dimensions do not conform");
            }
        }

        /// <summary>
        /// Adds Tensor A to this Tensor.
        /// </summary>
        public void Add(Tensor A)
        {
            checkShapeEqual(A);
            for (int i = 0; i < Shape[0]; i++)
            {
                data[i].Add(A.GetItem(i));
            }
        }

        /// <summary>
        /// Adds Matrix A to this Tensor.
        /// </summary>
        public void Add(Matrix A)
        {
            checkShapeEqual(A);
            for (int i = 0; i < Shape[0]; i++)
            {
                data[i].Add(A);
            }
        }

        /// <summary>
        /// Performs a Hadamard (elementwise) multiplication on this Tensor.
        /// </summary>
        public void Hadamard(Tensor A)
        {
            checkShapeEqual(A);
            for (int i = 0; i < Shape[0]; i++)
            {
                data[i].Hadamard(A.GetItem(i));
            }
        }

        /// <summary>
        /// Performs a Hadamard (elementwise) multiplication on this Tensor.
        /// </summary>
        public void Hadamard(Matrix A)
        {
            checkShapeEqual(A);
            for (int i = 0; i < Shape[0]; i++)
            {
                data[i].Hadamard(A);
            }
        }

        /// <summary>
        /// Calculates the dot product between the matricies contained in two tensors.
        /// </summary>
        /// <returns>An array of matrix dot products</returns>
        public float[] Dot(Tensor A)
        {
            checkShapeEqual(A);
            var dotp = new float[Shape[0]];
            for (int i = 0; i < Shape[0]; i++)
            {
                dotp[i] = data[i].Dot(A.GetItem(i));
            }
            return dotp;
        }

        /// <summary>
        /// Calculates the dot product between the matricies contained in this tensor and matrix A.
        /// </summary>
        /// <returns>An array of matrix dot products.</returns>
        public float[] Dot(Matrix A)
        {
            checkShapeEqual(A);
            var dotp = new float[Shape[0]];
            for (int i = 0; i < Shape[0]; i++)
            {
                dotp[i] = data[i].Dot(A);
            }
            return dotp;
        }

        /// <summary>
        /// Performs a matrix transformation on the matricies contained in this tensor.
        /// </summary>
        public void Transform(Tensor A)
        {
            if (Shape[0] == A.Shape[0]) // Subsequent matrix dimension validation is implemented in Matrix.Transform()
            {
                for (int i = 0; i < Shape[0]; i++)
                {
                    data[i].Transform(A.GetItem(i));
                }
            }
        }

        /// <summary>
        /// Performs a matrix transformation on the matricies contained in this tensor.
        /// </summary>
        public void Transform(Matrix A)
        {
            for (int i = 0; i < Shape[0]; i++)
            {
                data[i].Transform(A);
            }
        }

        /// <summary>
        /// Reshapes every matrix in the Tensor to the specified dimensions.
        /// </summary>
        public void Reshape(int rows, int cols)
        {
            if (Shape[1] * Shape[2] != rows * cols)
            {
                throw new Exception("Tensor does not fit reshape dimensions");
            }
            Shape[1] = rows;
            Shape[2] = cols;
            for (int i = 0; i < Shape[0]; i++)
            {
                data[i].Reshape(rows, cols);
            }
        }

        /// <summary>
        /// Transposes every matrix in the Tensor.
        /// </summary>
        public void Transpose()
        {
            for (int i = 0; i < Shape[0]; i++)
            {
                data[i].Transpose();
            }
        }

        /// <summary>
        /// Returns the tensor as an array of 2d float arrays.
        /// </summary>
        public float[][,] ToArray()
        {
            var tensorArray = new float[Shape[0]][,];
            for (int i = 0; i < Shape[0]; i++)
            {
                tensorArray[i] = data[i].ToArray();
            }
            return tensorArray;
        }
    }
}