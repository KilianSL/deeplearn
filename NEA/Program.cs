using System;
using NEA.Tensor;

namespace NEA
{
    class Program
    {
        static void Main(string[] args)
        {
            var matA = Matrix.GaussianMatrix(3, 3);
            Console.WriteLine(matA);
            Console.WriteLine();
            var matB = Matrix.GaussianMatrix(3, 3);
            Console.WriteLine(matB);
            Console.WriteLine();
            matA.Add(matB);
            Console.WriteLine(matA);
            Console.WriteLine();
        }
    }
}
