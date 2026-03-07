import unittest

class BasicTests(unittest.TestCase):
    def test_math(self):
        # *测试基础数学运算*
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()
