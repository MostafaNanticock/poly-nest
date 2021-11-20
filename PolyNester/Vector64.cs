using System;
using System.Collections.Generic;
using System.Text;

namespace PolyNester
{
    public struct Vector64
    {
        public double X;
        public double Y;
        public Vector64(double X, double Y)
        {
            this.X = X; this.Y = Y;
        }

        public static Vector64 operator +(Vector64 a, Vector64 b)
        {
            return new Vector64(a.X + b.X, a.Y + b.Y);
        }

        public static Vector64 operator *(Vector64 a, Vector64 b)
        {
            return new Vector64(a.X * b.X, a.Y * b.Y);
        }

        public static Vector64 operator *(double b, Vector64 a)
        {
            return new Vector64(a.X * b, a.Y * b);
        }
    }
}
