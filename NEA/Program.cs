using NEA.Tensor;
using System;

namespace NEA
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var matA = Matrix.GaussianMatrix(2, 3);
            foreach (var item in matA)
            {
                Console.WriteLine(item);
            }
        }
    }
}