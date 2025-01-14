﻿using ClipperLib;
using System;
using System.Collections;
using System.Collections.Generic;

namespace PolyNester
{
    /// <summary>Simply a list of points</summary>
    using NPolygon = List<IntPoint>;
    /// <summary>A simple list of Ngons</summary>
    using NPolygonsList = List<List<IntPoint>>;

    public static class GeomUtility
    {
        private class PolarComparer : IComparer
        {
            public static int CompareIntPoint(IntPoint A, IntPoint B)
            {
                long det = A.Y * B.X - A.X * B.Y;

                if (det == 0)
                {
                    long dot = A.X * B.X + A.Y * B.Y;
                    if (dot >= 0)
                        return 0;
                }

                if (A.Y == 0 && A.X > 0)
                    return -1;
                if (B.Y == 0 && B.X > 0)
                    return 1;
                if (A.Y > 0 && B.Y < 0)
                    return -1;
                if (A.Y < 0 && B.Y > 0)
                    return 1;
                return det > 0 ? 1 : -1;
            }

            int IComparer.Compare(object a, object b)
            {
                IntPoint A = (IntPoint)a;
                IntPoint B = (IntPoint)b;

                return CompareIntPoint(A, B);
            }
        }

        public static long Width(this IntRect rect)
        {
            return Math.Abs(rect.left - rect.right);
        }

        public static long Height(this IntRect rect)
        {
            return Math.Abs(rect.top - rect.bottom);
        }

        public static long Area(this IntRect rect)
        {
            return rect.Width() * rect.Height();
        }

        public static double Aspect(this IntRect rect)
        {
            return ((double)rect.Width()) / rect.Height();
        }

        public static NPolygon Clone(this NPolygon poly)
        {
            return new NPolygon(poly);
        }

        public static NPolygon Clone(this NPolygon poly, long shift_x, long shift_y, bool flip_first = false)
        {
            long scale = flip_first ? -1 : 1;

            NPolygon clone = new NPolygon(poly.Count);
            for (int i = 0; i < poly.Count; i++)
                clone.Add(new IntPoint(scale * poly[i].X + shift_x, scale * poly[i].Y + shift_y));
            return clone;
        }

        public static NPolygon Clone(this NPolygon poly, Mat3x3 T)
        {
            NPolygon clone = new NPolygon(poly.Count);
            for (int i = 0; i < poly.Count; i++)
                clone.Add(T * poly[i]);
            return clone;
        }

        public static NPolygonsList Clone(this NPolygonsList polys, long shift_x, long shift_y, bool flip_first = false)
        {
            NPolygonsList clone = new NPolygonsList(polys.Count);
            for (int i = 0; i < polys.Count; i++)
                clone.Add(polys[i].Clone(shift_x, shift_y, flip_first));
            return clone;
        }

        public static IntRect GetBounds(IEnumerable<IntPoint> points)
        {
            long width_min = long.MaxValue;
            long width_max = long.MinValue;
            long height_min = long.MaxValue;
            long height_max = long.MinValue;

            foreach (IntPoint p in points)
            {
                width_min = Math.Min(width_min, p.X);
                height_min = Math.Min(height_min, p.Y);
                width_max = Math.Max(width_max, p.X);
                height_max = Math.Max(height_max, p.Y);
            }

            return new IntRect(width_min, height_max, width_max, height_min);
        }

        public static Rect64 GetBounds(IEnumerable<Vector64> points)
        {
            double width_min = double.MaxValue;
            double width_max = double.MinValue;
            double height_min = double.MaxValue;
            double height_max = double.MinValue;

            foreach (Vector64 p in points)
            {
                width_min = Math.Min(width_min, p.X);
                height_min = Math.Min(height_min, p.Y);
                width_max = Math.Max(width_max, p.X);
                height_max = Math.Max(height_max, p.Y);
            }

            return new Rect64(width_min, height_max, width_max, height_min);
        }

        /// <to be Ported>
        public static void GetRefitTransform(IEnumerable<Vector64> points, Rect64 target, bool stretch, out Vector64 scale, out Vector64 shift)
        {
            Rect64 bds = GetBounds(points);

            scale = new Vector64(target.Width() / bds.Width(), target.Height() / bds.Height());

            if (!stretch)
            {
                double s = Math.Min(scale.X, scale.Y);
                scale = new Vector64(s, s);
            }

            shift = new Vector64(-bds.left, -bds.bottom) * scale
                + new Vector64(Math.Min(target.left, target.right), Math.Min(target.bottom, target.top));
        }

        /// <to be Ported>
        public static NPolygon ConvexHull(NPolygon subject, double rigidness = 0)
        {
            if (subject.Count == 0)
                return new NPolygon();

            if (rigidness >= 1)
                return subject.Clone();

            subject = subject.Clone();
            if (Clipper.Area(subject) < 0)
                Clipper.ReversePaths(new NPolygonsList() { subject });

            NPolygon last_hull = new NPolygon();
            NPolygon hull = subject;

            double subj_area = Clipper.Area(hull);

            int last_vert = 0;
            for (int i = 1; i < subject.Count; i++)
                if (hull[last_vert].Y > hull[i].Y)
                    last_vert = i;

            while (last_hull.Count != hull.Count)
            {
                last_hull = hull;
                hull = new NPolygon();
                hull.Add(last_hull[last_vert]);

                int steps_since_insert = 0;
                int max_steps = rigidness <= 0 ? int.MaxValue : (int)Math.Round(10 - (10 * rigidness));

                int n = last_hull.Count;

                int start = last_vert;
                for (int i = 1; i < n; i++)
                {
                    IntPoint a = last_hull[last_vert];
                    IntPoint b = last_hull[(start + i) % n];
                    IntPoint c = last_hull[(start + i + 1) % n];

                    IntPoint ab = new IntPoint(b.X - a.X, b.Y - a.Y);
                    IntPoint ac = new IntPoint(c.X - a.X, c.Y - a.Y);

                    if (ab.Y * ac.X < ab.X * ac.Y || steps_since_insert >= max_steps)
                    {
                        hull.Add(b);
                        last_vert = (start + i) % n;
                        steps_since_insert = -1;
                    }
                    steps_since_insert++;
                }

                last_vert = 0;

                double hull_area = Clipper.Area(hull);

                if (subj_area / hull_area < Math.Sqrt(rigidness))
                {
                    hull = Clipper.SimplifyPolygon(hull, PolyFillType.pftNonZero)[0];
                    break;
                }
            }

            return hull;
        }

        /// <to be Ported>
        public static NPolygonsList MinkowskiSumSegment(NPolygon pattern, IntPoint p1, IntPoint p2, bool flip_pattern)
        {
            Clipper clipper = new Clipper();

            NPolygon p1_c = pattern.Clone(p1.X, p1.Y, flip_pattern);

            if (p1 == p2)
                return new NPolygonsList() { p1_c };

            NPolygon p2_c = pattern.Clone(p2.X, p2.Y, flip_pattern);

            NPolygonsList full = new NPolygonsList();
            clipper.AddPath(p1_c, PolyType.ptSubject, true);
            clipper.AddPath(p2_c, PolyType.ptSubject, true);
            clipper.AddPaths(Clipper.MinkowskiSum(pattern.Clone(0, 0, flip_pattern), new NPolygon() { p1, p2 }, false), PolyType.ptSubject, true);
            clipper.Execute(ClipType.ctUnion, full, PolyFillType.pftNonZero);

            return full;
        }

        /// <to be Ported>
        public static NPolygonsList MinkowskiSumBoundary(NPolygon pattern, NPolygon path, bool flip_pattern)
        {
            Clipper clipper = new Clipper();

            NPolygonsList full = new NPolygonsList();

            for (int i = 0; i < path.Count; i++)
            {
                IntPoint p1 = path[i];
                IntPoint p2 = path[(i + 1) % path.Count];

                NPolygonsList seg = MinkowskiSumSegment(pattern, p1, p2, flip_pattern);
                clipper.AddPaths(full, PolyType.ptSubject, true);
                clipper.AddPaths(seg, PolyType.ptSubject, true);

                NPolygonsList res = new NPolygonsList();
                clipper.Execute(ClipType.ctUnion, res, PolyFillType.pftNonZero);
                full = res;
                clipper.Clear();
            }

            return full;
        }

        /// <to be Ported>
        public static NPolygonsList MinkowskiSumBoundary(NPolygon pattern, NPolygonsList path, bool flip_pattern)
        {
            Clipper clipper = new Clipper();

            NPolygonsList full = new NPolygonsList();

            for (int i = 0; i < path.Count; i++)
            {
                NPolygonsList seg = MinkowskiSumBoundary(pattern, path[i], flip_pattern);
                clipper.AddPaths(full, PolyType.ptSubject, true);
                clipper.AddPaths(seg, PolyType.ptSubject, true);

                NPolygonsList res = new NPolygonsList();
                clipper.Execute(ClipType.ctUnion, res, PolyFillType.pftNonZero);
                full = res;
                clipper.Clear();
            }

            return full;
        }

        /// <to be Ported>
        private static NPolygonsList MSumSimple(NPolygon pattern, NPolygonsList subject, bool flip_pattern)
        {
            IntRect pB = GetBounds(pattern);
            IntRect sB = GetBounds(subject[0]);

            if (flip_pattern)
            {
                pB = new IntRect(-pB.right, -pB.bottom, -pB.left, -pB.top);
            }

            long l = pB.left + sB.left;
            long r = pB.right + sB.right;
            long t = pB.top + sB.top;
            long b = pB.bottom + sB.bottom;

            NPolygon p = new NPolygon() { new IntPoint(l, b), new IntPoint(r, b), new IntPoint(r, t), new IntPoint(l, t) };
            return new NPolygonsList() { p };
        }

        /// <to be Ported>
        private static NPolygonsList MSumConvex(NPolygon pattern, NPolygonsList subject, bool flip_pattern)
        {
            NPolygon h_p = ConvexHull(pattern.Clone(0, 0, flip_pattern));
            NPolygon h_s = ConvexHull(subject[0].Clone());

            int n_p = h_p.Count;
            int n_s = h_s.Count;

            int sp = 0;
            for (int k = 0; k < n_p; k++)
                if (h_p[k].Y < h_p[sp].Y)
                    sp = k;

            int ss = 0;
            for (int k = 0; k < n_s; k++)
                if (h_s[k].Y < h_s[ss].Y)
                    ss = k;

            NPolygon poly = new NPolygon(n_p + n_s);

            int i = 0;
            int j = 0;
            while (i < n_p || j < n_s)
            {
                int ip = (sp + i + 1) % n_p;
                int jp = (ss + j + 1) % n_s;
                int ii = (sp + i) % n_p;
                int jj = (ss + j) % n_s;

                IntPoint sum = new IntPoint(h_p[ii].X + h_s[jj].X, h_p[ii].Y + h_s[jj].Y);
                IntPoint v = new IntPoint(h_p[ip].X - h_p[ii].X, h_p[ip].Y - h_p[ii].Y);
                IntPoint w = new IntPoint(h_s[jp].X - h_s[jj].X, h_s[jp].Y - h_s[jj].Y);

                poly.Add(sum);

                if (i == n_p)
                {
                    j++;
                    continue;
                }

                if (j == n_s)
                {
                    i++;
                    continue;
                }

                long cross = v.Y * w.X - v.X * w.Y;

                if (cross < 0) i++;
                else if (cross > 0) j++;
                else
                {
                    long dot = v.X * w.X + v.Y * w.Y;
                    if (dot > 0)
                    {
                        i++;
                        j++;
                    }
                    else
                    {
                        throw new Exception();
                    }
                }
            }

            return Clipper.SimplifyPolygon(poly);
        }

        /// <to be Ported>
        private static NPolygonsList MSumConcave(NPolygon pattern, NPolygonsList subject, bool flip_pattern, double rigidness = 1.0)
        {
            NPolygon subj = subject[0];
            NPolygon patt = pattern.Clone(0, 0, flip_pattern);

            if (rigidness < 1.0)
            {
                subj = ConvexHull(subj, rigidness);
                patt = ConvexHull(patt, rigidness);
            }

            NPolygonsList sres = MinkowskiSumBoundary(patt, subj, false);
            return sres.Count == 0 ? sres : new NPolygonsList() { sres[0] };
        }

        /// <to be Ported>
        private static NPolygonsList MSumFull(NPolygon pattern, NPolygonsList subject, bool flip_pattern)
        {
            Clipper clipper = new Clipper();

            NPolygonsList full = new NPolygonsList();

            long scale = flip_pattern ? -1 : 1;

            for (int i = 0; i < pattern.Count; i++)
                clipper.AddPaths(subject.Clone(scale * pattern[i].X, scale * pattern[i].Y), PolyType.ptSubject, true);

            clipper.Execute(ClipType.ctUnion, full, PolyFillType.pftNonZero);
            clipper.Clear();

            clipper.AddPaths(full, PolyType.ptSubject, true);
            clipper.AddPaths(MinkowskiSumBoundary(pattern, subject, flip_pattern), PolyType.ptSubject, true);

            NPolygonsList res = new NPolygonsList();

            clipper.Execute(ClipType.ctUnion, res, PolyFillType.pftNonZero);

            return res;
        }

        /// <to be Ported>
        public static NPolygonsList MinkowskiSum(NPolygon pattern, NPolygonsList subject, NFPQUALITY quality, bool flip_pattern)
        {
            switch (quality)
            {
                case NFPQUALITY.Simple:
                    return MSumSimple(pattern, subject, flip_pattern);
                case NFPQUALITY.Convex:
                    return MSumConvex(pattern, subject, flip_pattern);
                case NFPQUALITY.ConcaveLight:
                    return MSumConcave(pattern, subject, flip_pattern, 0.25);
                case NFPQUALITY.ConcaveMedium:
                    return MSumConcave(pattern, subject, flip_pattern, 0.55);
                case NFPQUALITY.ConcaveHigh:
                    return MSumConcave(pattern, subject, flip_pattern, 0.85);
                case NFPQUALITY.ConcaveFull:
                    return MSumConcave(pattern, subject, flip_pattern, 1.0);
                case NFPQUALITY.Full:
                    return MSumFull(pattern, subject, flip_pattern);
                default:
                    return null;
            }
        }

        /// <to be Ported>
        public static NPolygon CanFitInsidePolygon(IntRect canvas, NPolygon pattern)
        {
            IntRect bds = GetBounds(pattern);

            long l = canvas.left - bds.left;
            long r = canvas.right - bds.right;
            long t = canvas.top - bds.top;
            long b = canvas.bottom - bds.bottom;

            if (l > r || b > t)
                return null;

            return new NPolygon() { new IntPoint(l, b), new IntPoint(r, b), new IntPoint(r, t), new IntPoint(l, t) };
        }

        /// <to be Ported>
        public static double AlignToEdgeRotation(NPolygon target, int edge_start)
        {
            edge_start %= target.Count;
            int next_pt = (edge_start + 1) % target.Count;
            IntPoint best_edge = new IntPoint(target[next_pt].X - target[edge_start].X, target[next_pt].Y - target[edge_start].Y);
            return -Math.Atan2(best_edge.Y, best_edge.X);
        }

        /// <to be Ported>
        public static bool AlmostRectangle(NPolygon target, double percent_diff = 0.05)
        {
            IntRect bounds = GetBounds(target);
            double area = Math.Abs(Clipper.Area(target));

            return 1.0 - area / bounds.Area() < percent_diff;
        }
    }
}
