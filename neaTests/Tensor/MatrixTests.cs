using Microsoft.VisualStudio.TestTools.UnitTesting;
using NEA.Tensor;

namespace neaTests
{
    [TestClass]
    public class MatrixTests
    {
        // "Expected" Data for each test - 3x3 square matrix filled with 2s
        // Stored here as float[,] to ensure that the actual objects used in tests remain decoupled
        private float[,]  dataA = new float[,] { { 2, 2, 2 }, { 2, 2, 2 }, { 2, 2, 2 } };

        private float[,] dataB = new float[,] { { 2, 2, 2 }, { 2, 2, 2 }, { 2, 2, 2 } };

        [TestMethod]
        public void Add_ExpectedData_ReturnsSum()
        {
            // Arrange
            var expected = new Matrix(new float[,] { { 4, 4, 4 }, { 4, 4, 4 }, { 4, 4, 4 } });
            var matA = new Matrix(dataA);
            var matB = new Matrix(dataB);

            // Act
            matA.Add(matB);

            // Assert
            Assert.AreEqual(expected, matA);

        }
    }
}