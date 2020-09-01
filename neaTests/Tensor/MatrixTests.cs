using Microsoft.VisualStudio.TestTools.UnitTesting;
using NEA.Tensor;

namespace neaTests
{
    [TestClass]
    public class MatrixTests
    {
        // TestContext object for logging to the debug log within a test environment
        public TestContext TestContext { get; set; }

        // "Expected" Data for each test - 3x3 square matrix filled with 2s
        // Stored here as float[,] to ensure that the actual objects used in tests remain decoupled
        private float[,] dataA = new float[,] { { 2, 2, 2 }, { 2, 2, 2 }, { 2, 2, 2 } };

        private float[,] dataB = new float[,] { { 2, 2, 2 }, { 2, 2, 2 }, { 2, 2, 2 } };

        // Tests the equals method works - if this fails disregard all other unit tests
        [TestMethod]
        public void Equals_ExpectedData_ReturnsTrue()
        {
            var matA = new Matrix(dataA);
            var matB = new Matrix(dataB);

            Assert.IsTrue(matA.Equals(matB));
        }

        [TestMethod]
        public void Add_ExpectedData()
        {
            // Arrange
            var expected = new Matrix(new float[,] { { 4, 4, 4 }, { 4, 4, 4 }, { 4, 4, 4 } });
            var matA = new Matrix(dataA);
            var matB = new Matrix(dataB);

            // Act
            matA.Add(matB);

            TestContext.WriteLine(matA.ToString());

            // Assert
            Assert.AreEqual(expected, matA);
        }

        [TestMethod]
        public void HadamardProduct_ExpectedData()
        {
            var expected = new Matrix(new float[,] { { 4, 4, 4 }, { 4, 4, 4 }, { 4, 4, 4 } });
            var matA = new Matrix(dataA);
            var matB = new Matrix(dataB);

            matA.Hadamard(matB);

            Assert.AreEqual(expected, matA);
        }

        [TestMethod]
        public void DotProduct_ExpectedData()
        {
            var matA = new Matrix(dataA);
            var matB = new Matrix(dataB);

            var result = matA.Dot(matB);

            Assert.AreEqual(36f, result);
        }

        [TestMethod]
        public void MatrixTransform_ExpectedData()
        {
            var expected = new Matrix(new float[,] { { 12, 12, 12 }, { 12, 12, 12 }, { 12, 12, 12 } });
            var matA = new Matrix(dataA);
            var matB = new Matrix(dataB);

            matA.Transform(matB);

            TestContext.WriteLine(matA.ToString());

            Assert.AreEqual(expected, matA);
        }

        [TestMethod]
        public void MatrixReshape_ExpectedData_Returns9x1()
        {
            var expected = new Matrix(new float[,] { { 2 }, { 2 }, { 2 }, { 2 }, { 2 }, { 2 }, { 2 }, { 2 }, { 2 } });
            var matA = new Matrix(dataA);

            matA.Reshape(9, 1);

            TestContext.WriteLine(matA.ToString());

            Assert.AreEqual(expected, matA);
        }

        [TestMethod]
        public void MatrixTranspose_ExpectedData()
        {
            var expected = new Matrix(new float[,] { { 1, 3, 5 }, { 2, 4, 6 } });
            var matA = new Matrix(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

            matA.Transpose();

            TestContext.WriteLine(matA.ToString());

            Assert.AreEqual(expected, matA);
        }
    }
}