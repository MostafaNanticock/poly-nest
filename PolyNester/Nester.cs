/*
 * 
 * Only parts marked with "/// <to be Ported>" marker will be ported to cpp
 * 
 */

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using ClipperLib;

namespace PolyNester
{
    /// <summary>Simply a list of points</summary>
    using NPolygon = List<IntPoint>;
    /// <summary>A simple list of Ngons</summary>
    using NPolygonsList = List<List<IntPoint>>;

    public enum NFPQUALITY
    {
        Simple,
        Convex,
        ConcaveLight,
        ConcaveMedium,
        ConcaveHigh,
        ConcaveFull,
        Full
    }

    public class Nester
    {
        /// <to be Ported>
        private class PolyRef
        {
            public NPolygonsList poly;
            public Mat3x3 trans;

            /// <to be Ported>
            public IntPoint GetTransformedPoint(int poly_id, int index)
            {
                return trans * poly[poly_id][index];
            }

            /// <to be Ported>
            public NPolygonsList GetTransformedPoly()
            {
                NPolygonsList n = new NPolygonsList(poly.Count);
                for (int i = 0; i < poly.Count; i++)
                {
                    NPolygon nn = new NPolygon(poly[i].Count);
                    for (int j = 0; j < poly[i].Count; j++)
                        nn.Add(GetTransformedPoint(i, j));
                    n.Add(nn);
                }
                return n;
            }
        }

        /// <to be Ported>
        private class Command
        {
            public Action<object[]> Call;
            public object[] param;
        }

        private const long unit_scale = 10000000;

        private List<PolyRef> polygon_lib;  // list of saved polygons for reference by handle, stores raw poly positions and transforms

        private Queue<Command> command_buffer;  // buffers list of commands which will append transforms to elements of poly_lib on execute

        private BackgroundWorker background_worker;     // used to execute command buffer in background

        public int LibSize { get { return polygon_lib.Count; } }

        public Nester()
        {
            polygon_lib = new List<PolyRef>();
            command_buffer = new Queue<Command>();
        }

        /// <to be Ported>
        public void ExecuteCommandBuffer(Action<ProgressChangedEventArgs> callback_progress, Action<AsyncCompletedEventArgs> callback_completed)
        {
            background_worker = new BackgroundWorker();
            background_worker.WorkerSupportsCancellation = true;
            background_worker.WorkerReportsProgress = true;

            if (callback_progress != null)
                background_worker.ProgressChanged += (sender, e) => callback_progress.Invoke(e);
            if (callback_completed != null)
                background_worker.RunWorkerCompleted += (sender, e) => callback_completed.Invoke(e);

            background_worker.DoWork += Background_worker_DoWork;
            background_worker.RunWorkerCompleted += Background_worker_RunWorkerCompleted;

            background_worker.RunWorkerAsync();
        }

        public void CancelExecute()
        {
            if (!IsBusy())
                return;

            background_worker.CancelAsync();
        }

        /// <to be Ported>
        private void Background_worker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Cancelled || e.Error != null)
            {
                ResetTransformLib();
                command_buffer.Clear();
            }

            background_worker.Dispose();
        }

        /// <to be Ported>
        private void Background_worker_DoWork(object sender, DoWorkEventArgs e)
        {
            while (command_buffer.Count > 0)
            {
                Command cmd = command_buffer.Dequeue();
                cmd.Call(cmd.param);

                if (background_worker.CancellationPending)
                {
                    e.Cancel = true;
                    break;
                }
            }
        }

        /// <to be Ported>
        public void ClearCommandBuffer()
        {
            command_buffer.Clear();
        }

        public bool IsBusy()
        {
            return background_worker != null && background_worker.IsBusy;
        }

        /// <to be Ported>
        private HashSet<int> PreprocessHandles(IEnumerable<int> handles)
        {
            if (handles == null)
                handles = Enumerable.Range(0, polygon_lib.Count);

            HashSet<int> unique = new HashSet<int>();
            foreach (int i in handles)
                unique.Add(i);

            return unique;
        }

        /// <to be Ported>
        private void cmd_scale(params object[] param)
        {
            int handle = (int)param[0];
            double scale_x = (double)param[1];
            double scale_y = (double)param[2];
            polygon_lib[handle].trans = Mat3x3.Scale(scale_x, scale_y) * polygon_lib[handle].trans;
        }

        /// <to be Ported>
        private void cmd_rotate(params object[] param)
        {
            int handle = (int)param[0];
            double theta = (double)param[1];
            polygon_lib[handle].trans = Mat3x3.RotateCounterClockwise(theta) * polygon_lib[handle].trans;
        }

        /// <to be Ported>
        private void cmd_translate(params object[] param)
        {
            int handle = (int)param[0];
            double translate_x = (double)param[1];
            double translate_y = (double)param[2];
            polygon_lib[handle].trans = Mat3x3.Translate(translate_x, translate_y) * polygon_lib[handle].trans;
        }

        /// <to be Ported>
        private void cmd_translate_origin_to_zero(params object[] param)
        {
            HashSet<int> unique = PreprocessHandles(param[0] as IEnumerable<int>);

            foreach (int i in unique)
            {
                IntPoint o = polygon_lib[i].GetTransformedPoint(0, 0);
                cmd_translate(i, (double)-o.X, (double)-o.Y);
            }
        }

        /// <to be Ported>
        public void CMD_Refit(Rect64 target, bool stretch, IEnumerable<int> handles)
        {
            command_buffer.Enqueue(new Command() { Call = cmd_refit, param = new object[] { target, stretch, handles } });
        }

        /// <to be Ported>
        private void cmd_refit(params object[] param)
        {
            Rect64 target = (Rect64)param[0];
            bool stretch = (bool)param[1];
            HashSet<int> unique = PreprocessHandles(param[2] as IEnumerable<int>);

            HashSet<Vector64> points = new HashSet<Vector64>();
            foreach (int i in unique)
                points.UnionWith(polygon_lib[i].poly[0].Select(p => polygon_lib[i].trans * new Vector64(p.X, p.Y)));

            Vector64 scale, trans;
            GeomUtility.GetRefitTransform(points, target, stretch, out scale, out trans);

            foreach (int i in unique)
            {
                cmd_scale(i, scale.X, scale.Y);
                cmd_translate(i, trans.X, trans.Y);
            }
        }

        /// <to be Ported>
        /// <summary>
        /// Get the optimal quality for tradeoff between speed and precision of NFP
        /// </summary>
        /// <param name="subj_handle"></param>
        /// <param name="pattern_handle"></param>
        /// <returns></returns>
        private NFPQUALITY GetNFPQuality(int subj_handle, int pattern_handle, double max_area_bounds)
        {
            NPolygon S = polygon_lib[subj_handle].GetTransformedPoly()[0];
            NPolygon P = polygon_lib[pattern_handle].GetTransformedPoly()[0];

            if (GeomUtility.AlmostRectangle(S) && GeomUtility.AlmostRectangle(P))
                return NFPQUALITY.Simple;

            double s_A = GeomUtility.GetBounds(S).Area();
            double p_A = GeomUtility.GetBounds(P).Area();

            if (p_A / s_A > 1000)
                return NFPQUALITY.Simple;

            if (s_A / max_area_bounds < 0.05)
                return NFPQUALITY.Simple;

            if (p_A / s_A > 100)
                return NFPQUALITY.Convex;

            if (p_A / s_A > 50)
                return NFPQUALITY.ConcaveLight;

            if (p_A / s_A > 10)
                return NFPQUALITY.ConcaveMedium;

            if (p_A / s_A > 2)
                return NFPQUALITY.ConcaveHigh;

            if (p_A / s_A > 0.25)
                return NFPQUALITY.ConcaveFull;

            return NFPQUALITY.Full;
        }

        /// <to be Ported>
        /// <summary>
        /// Parallel kernel for generating NFP of pattern on handle, return the index in the library of this NFP
        /// Decides the optimal quality for this NFP
        /// </summary>
        /// <param name="subj_handle"></param>
        /// <param name="pattern_handle"></param>
        /// <param name="lib_set_at"></param>
        /// <returns></returns>
        private int NFPKernel(int subj_handle, int pattern_handle, double max_area_bounds, int lib_set_at, NFPQUALITY max_quality = NFPQUALITY.Full)
        {
            NFPQUALITY quality = GetNFPQuality(subj_handle, pattern_handle, max_area_bounds);
            quality = (NFPQUALITY)Math.Min((int)quality, (int)max_quality);
            return AddMinkowskiSum(subj_handle, pattern_handle, quality, true, lib_set_at);
        }

        /// <to be Ported>
        public void CMD_Nest(IEnumerable<int> handles, NFPQUALITY max_quality = NFPQUALITY.Full)
        {
            command_buffer.Enqueue(new Command() { Call = cmd_nest, param = new object[] { handles, max_quality } });
        }

        /// <to be Ported>
        /// <summary>
        /// Nest the collection of handles with minimal enclosing square from origin
        /// </summary>
        /// <param name="handles"></param>
        private void cmd_nest(params object[] param)
        {
            HashSet<int> unique = PreprocessHandles(param[0] as IEnumerable<int>);
            NFPQUALITY max_quality = (NFPQUALITY)param[1];

            cmd_translate_origin_to_zero(unique);

            int n = unique.Count;

            Dictionary<int, IntRect> bounds = new Dictionary<int, IntRect>();
            foreach (int handle in unique)
                bounds.Add(handle, GeomUtility.GetBounds(polygon_lib[handle].GetTransformedPoly()[0]));

            int[] ordered_handles = unique.OrderByDescending(p => Math.Max(bounds[p].Height(), bounds[p].Width())).ToArray();
            double max_bound_area = bounds[ordered_handles[0]].Area();

            int start_cnt = polygon_lib.Count;

            int[] canvas_regions = AddCanvasFitPolygon(ordered_handles);

            int base_cnt = polygon_lib.Count;
            for (int i = 0; i < n * n - n; i++)
                polygon_lib.Add(new PolyRef());

            int update_breaks = 10;
            int nfp_chunk_sz = n * n / update_breaks * update_breaks == n * n ? n * n / update_breaks : n * n / update_breaks + 1;

            // the row corresponds to pattern and col to nfp for this pattern on col subj
            int[,] nfps = new int[n, n];
            for (int k = 0; k < update_breaks; k++)
            {
                int start = k * nfp_chunk_sz;
                int end = Math.Min((k + 1) * nfp_chunk_sz, n * n);

                if (start >= end)
                    break;

                Parallel.For(start, end, i => nfps[i / n, i % n] = i / n == i % n ? -1 : NFPKernel(ordered_handles[i % n], ordered_handles[i / n], max_bound_area, base_cnt + i - (i % n > i / n ? 1 : 0) - i / n, max_quality));

                double progress = Math.Min(((double)(k + 1)) / (update_breaks + 1) * 50.0, 50.0);
                background_worker.ReportProgress((int)progress);

                if (background_worker.CancellationPending)
                    break;
            }

            int place_chunk_sz = Math.Max(n / update_breaks, 1);

            bool[] placed = new bool[n];
            for (int i = 0; i < n; i++)
            {
                if (i % 10 == 0 && background_worker.CancellationPending)
                    break;

                Clipper c = new Clipper();
                c.AddPath(polygon_lib[canvas_regions[i]].poly[0], PolyType.ptSubject, true);
                for (int j = 0; j < i; j++)
                {
                    if (!placed[j])
                        continue;

                    c.AddPaths(polygon_lib[nfps[i, j]].GetTransformedPoly(), PolyType.ptClip, true);
                }
                NPolygonsList fit_region = new NPolygonsList();
                c.Execute(ClipType.ctDifference, fit_region, PolyFillType.pftNonZero);


                IntPoint o = polygon_lib[ordered_handles[i]].GetTransformedPoint(0, 0);
                IntRect bds = bounds[ordered_handles[i]];
                long ext_x = bds.right - o.X;
                long ext_y = bds.top - o.Y;
                IntPoint place = new IntPoint(0, 0);
                long pl_score = long.MaxValue;
                for (int k = 0; k < fit_region.Count; k++)
                    for (int l = 0; l < fit_region[k].Count; l++)
                    {
                        IntPoint cand = fit_region[k][l];
                        long cd_score = Math.Max(cand.X + ext_x, cand.Y + ext_y);
                        if (cd_score < pl_score)
                        {
                            pl_score = cd_score;
                            place = cand;
                            placed[i] = true;
                        }
                    }

                if (!placed[i])
                    continue;

                cmd_translate(ordered_handles[i], (double)(place.X - o.X), (double)(place.Y - o.Y));
                for (int k = i + 1; k < n; k++)
                    cmd_translate(nfps[k, i], (double)(place.X - o.X), (double)(place.Y - o.Y));

                if (i % place_chunk_sz == 0)
                {
                    double progress = Math.Min(60.0 + ((double)(i / place_chunk_sz)) / (update_breaks + 1) * 40.0, 100.0);
                    background_worker.ReportProgress((int)progress);
                }
            }

            // remove temporary added values
            polygon_lib.RemoveRange(start_cnt, polygon_lib.Count - start_cnt);
        }

        /// <to be Ported>
        public void CMD_OptimalRotation(IEnumerable<int> handles)
        {
            command_buffer.Enqueue(new Command() { Call = cmd_optimal_rotation, param = new object[] { handles } });
        }

        /// <to be Ported>
        private void cmd_optimal_rotation(int handle)
        {
            NPolygon hull = polygon_lib[handle].GetTransformedPoly()[0];
            int n = hull.Count;

            double best_t = 0;
            int best = 0;
            long best_area = long.MaxValue;
            bool flip_best = false;

            for (int i = 0; i < n; i++)
            {
                double t = GeomUtility.AlignToEdgeRotation(hull, i);

                Mat3x3 rot = Mat3x3.RotateCounterClockwise(t);

                NPolygon clone = hull.Clone(rot);

                IntRect bounds = GeomUtility.GetBounds(clone);
                long area = bounds.Area();
                double aspect = bounds.Aspect();

                if (area < best_area)
                {
                    best_area = area;
                    best = i;
                    best_t = t;
                    flip_best = aspect > 1.0;
                }
            }

            double flip = flip_best ? Math.PI * 0.5 : 0;
            IntPoint around = hull[best];

            cmd_translate(handle, (double)-around.X, (double)-around.Y);
            cmd_rotate(handle, best_t + flip);
            cmd_translate(handle, (double)around.X, (double)around.Y);
        }

        /// <to be Ported>
        private void cmd_optimal_rotation(params object[] param)
        {
            HashSet<int> unique = PreprocessHandles(param[0] as IEnumerable<int>);

            foreach (int i in unique)
                cmd_optimal_rotation(i);
        }

        /// <to be Ported>
        /// <summary>
        /// Append a set triangulated polygons to the nester and get handles for each point to the correp. polygon island
        /// </summary>
        /// <param name="points"></param>
        /// <param name="tris"></param>
        /// <returns></returns>
        public int[] AddPolygons(IntPoint[] points, int[] tris, double miter_distance = 0.0)
        {
            // from points to clusters of tris
            int[] poly_map = new int[points.Length];
            for (int i = 0; i < poly_map.Length; i++)
                poly_map[i] = -1;

            HashSet<int>[] graph = new HashSet<int>[points.Length];
            for (int i = 0; i < graph.Length; i++)
                graph[i] = new HashSet<int>();

            for (int i = 0; i < tris.Length; i += 3)
            {
                int t1 = tris[i];
                int t2 = tris[i + 1];
                int t3 = tris[i + 2];

                graph[t1].Add(t2);
                graph[t1].Add(t3);
                graph[t2].Add(t1);
                graph[t2].Add(t3);
                graph[t3].Add(t1);
                graph[t3].Add(t2);
            }

            if (graph.Any(p => p.Count == 0))
                throw new Exception("No singular vertices should exist on mesh");

            int[] clust_ids = new int[points.Length];

            HashSet<int> unmarked = new HashSet<int>(Enumerable.Range(0, points.Length));
            int clust_cnt = 0;
            while (unmarked.Count > 0)
            {
                Queue<int> open = new Queue<int>();
                int first = unmarked.First();
                unmarked.Remove(first);
                open.Enqueue(first);
                while (open.Count > 0)
                {
                    int c = open.Dequeue();
                    clust_ids[c] = clust_cnt;
                    foreach (int n in graph[c])
                    {
                        if (unmarked.Contains(n))
                        {
                            unmarked.Remove(n);
                            open.Enqueue(n);
                        }
                    }
                }

                clust_cnt++;
            }

            NPolygonsList[] clusters = new NPolygonsList[clust_cnt];
            for (int i = 0; i < tris.Length; i += 3)
            {
                int clust = clust_ids[tris[i]];
                if (clusters[clust] == null)
                    clusters[clust] = new NPolygonsList();

                IntPoint p1 = points[tris[i]];
                IntPoint p2 = points[tris[i + 1]];
                IntPoint p3 = points[tris[i + 2]];

                clusters[clust].Add(new NPolygon() { p1, p2, p3 });
            }

            List<NPolygonsList> fulls = new List<NPolygonsList>();

            for (int i = 0; i < clust_cnt; i++)
            {
                NPolygonsList cl = clusters[i];

                Clipper c = new Clipper();
                foreach (NPolygon n in cl)
                    c.AddPath(n, PolyType.ptSubject, true);

                NPolygonsList full = new NPolygonsList();
                c.Execute(ClipType.ctUnion, full, PolyFillType.pftNonZero);
                full = Clipper.SimplifyPolygons(full, PolyFillType.pftNonZero);

                if (miter_distance > 0.00001)
                {
                    NPolygonsList full_miter = new NPolygonsList();
                    ClipperOffset co = new ClipperOffset();
                    co.AddPaths(full, JoinType.jtMiter, EndType.etClosedPolygon);
                    co.Execute(ref full_miter, miter_distance);
                    full_miter = Clipper.SimplifyPolygons(full_miter, PolyFillType.pftNonZero);
                    fulls.Add(full_miter);
                }
                else
                    fulls.Add(full);
            }

            for (int i = 0; i < clust_ids.Length; i++)
                clust_ids[i] += polygon_lib.Count;

            for (int i = 0; i < fulls.Count; i++)
                polygon_lib.Add(new PolyRef() { poly = fulls[i], trans = Mat3x3.Eye() });

            return clust_ids;
        }

        /// <to be Ported>
        /// <summary>
        /// Append a set of triangulated polygons to the nester and get handles for points to polygons, coordinates are assumed
        /// to be in UV [0,1] space
        /// </summary>
        /// <param name="points"></param>
        /// <param name="tris"></param>
        /// <returns></returns>
        public int[] AddUVPolygons(Vector64[] points, int[] tris, double miter_distance = 0.0)
        {
            IntPoint[] new_pts = new IntPoint[points.Length];
            for (int i = 0; i < points.Length; i++)
                new_pts[i] = new IntPoint(points[i].X * unit_scale, points[i].Y * unit_scale);

            int[] map = AddPolygons(new_pts, tris, miter_distance * unit_scale);

            return map;
        }

        /// <to be Ported>
        public int AddMinkowskiSum(int subj_handle, int pattern_handle, NFPQUALITY quality, bool flip_pattern, int set_at = -1)
        {
            NPolygonsList A = polygon_lib[subj_handle].GetTransformedPoly();
            NPolygonsList B = polygon_lib[pattern_handle].GetTransformedPoly();

            NPolygonsList C = GeomUtility.MinkowskiSum(B[0], A, quality, flip_pattern);
            PolyRef pref = new PolyRef() { poly = C, trans = Mat3x3.Eye() };

            if (set_at < 0)
                polygon_lib.Add(pref);
            else
                polygon_lib[set_at] = pref;

            return set_at < 0 ? polygon_lib.Count - 1 : set_at;
        }

        /// <to be Ported>
        public int AddCanvasFitPolygon(IntRect canvas, int pattern_handle)
        {
            NPolygon B = polygon_lib[pattern_handle].GetTransformedPoly()[0];

            NPolygon C = GeomUtility.CanFitInsidePolygon(canvas, B);
            polygon_lib.Add(new PolyRef() { poly = new NPolygonsList() { C }, trans = Mat3x3.Eye() });
            return polygon_lib.Count - 1;
        }

        /// <to be Ported>
        public int[] AddCanvasFitPolygon(IEnumerable<int> handles)
        {
            HashSet<int> unique = PreprocessHandles(handles);

            long w = 0;
            long h = 0;

            foreach (int i in unique)
            {
                IntRect bds = GeomUtility.GetBounds(polygon_lib[i].GetTransformedPoly()[0]);
                w += bds.Width();
                h += bds.Height();
            }

            w += 1000;
            h += 1000;

            IntRect canvas = new IntRect(0, h, w, 0);

            return handles.Select(p => AddCanvasFitPolygon(canvas, p)).ToArray();
        }

        /// <to be Ported>
        public NPolygonsList GetTransformedPoly(int handle)
        {
            return polygon_lib[handle].GetTransformedPoly();
        }

        /// <to be Ported>
        public void ResetTransformLib()
        {
            for (int i = 0; i < polygon_lib.Count; i++)
                polygon_lib[i].trans = Mat3x3.Eye();
        }
    }
}
