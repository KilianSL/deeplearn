using NEA.Tensor;
using System;

namespace NEA
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var matA = Matrix.GaussianMatrix(2, 3);
            Console.WriteLine(matA);
            Console.WriteLine();
            var matB = Matrix.GaussianMatrix(2, 3);
            Console.WriteLine(matB);
            Console.WriteLine();
            matA.Add(matB);
            Console.WriteLine(matA);
            Console.WriteLine();
            matA.Reshape(6, 1);
            Console.WriteLine(matA);
        }
    }
}