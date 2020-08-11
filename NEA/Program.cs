using System;

namespace NEA
{
    class Program
    {
        static void Main(string[] args)
        {
            var matrix = Tensor.Matrix.GaussianMatrix(5, 5);
            foreach (var item in matrix.data)
            {
                Console.WriteLine(item);
            }
        }
    }
}
