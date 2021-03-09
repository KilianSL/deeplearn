using System;
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
        public float this[int idx]
        {
            get
            {
                if (Shape[0] == 1)
                {
                    return data[0][idx]; // will handle error in the case where the matrix is not an Nx1 vector
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
                    data[0][idx] = value; // will handle error in the case where the matrix is not an Nx1 vector
                }
                else
                {
                    throw new Exception("Unsuitable Tensor Indexing");
                }
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
    }
}