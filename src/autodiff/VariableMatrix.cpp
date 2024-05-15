// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/VariableMatrix.hpp"

#include <Eigen/QR>

namespace sleipnir {

VariableMatrix Solve(const VariableMatrix& A, const VariableMatrix& B) {
  // m x n * n x p = m x p
  Assert(A.Rows() == B.Rows());

  if (A.Rows() == 1 && A.Cols() == 1) {
    // Compute optimal inverse instead of using Eigen's general solver
    return B[0, 0] / A[0, 0];
  } else if (A.Rows() == 2 && A.Cols() == 2) {
    // Compute optimal inverse instead of using Eigen's general solver
    //
    // [a  b]⁻¹  ___1___ [ d  −b]
    // [c  d]  = ad − bc [−c   a]

    const auto& a = A[0, 0];
    const auto& b = A[0, 1];
    const auto& c = A[1, 0];
    const auto& d = A[1, 1];

    sleipnir::VariableMatrix adjA{{d, -b}, {-c, a}};
    auto detA = a * d - b * c;
    return adjA / detA * B;
  } else if (A.Rows() == 3 && A.Cols() == 3) {
    // Compute optimal inverse instead of using Eigen's general solver
    //
    // [a  b  c]⁻¹
    // [d  e  f]
    // [g  h  i]
    //                        1                   [ei − fh  ch − bi  bf − ce]
    //     = ------------------------------------ [fg − di  ai − cg  cd − af]
    //       a(ei − fh) + b(fg − di) + c(dh − eg) [dh − eg  bg − ah  ae − bd]
    //
    // https://www.wolframalpha.com/input?i=inverse+%7B%7Ba%2C+b%2C+c%7D%2C+%7Bd%2C+e%2C+f%7D%2C+%7Bg%2C+h%2C+i%7D%7D

    const auto& a = A[0, 0];
    const auto& b = A[0, 1];
    const auto& c = A[0, 2];
    const auto& d = A[1, 0];
    const auto& e = A[1, 1];
    const auto& f = A[1, 2];
    const auto& g = A[2, 0];
    const auto& h = A[2, 1];
    const auto& i = A[2, 2];

    auto ae = a * e;
    auto af = a * f;
    auto ah = a * h;
    auto ai = a * i;
    auto bd = b * d;
    auto bf = b * f;
    auto bg = b * g;
    auto bi = b * i;
    auto cd = c * d;
    auto ce = c * e;
    auto cg = c * g;
    auto ch = c * h;
    auto dh = d * h;
    auto di = d * i;
    auto eg = e * g;
    auto ei = e * i;
    auto fg = f * g;
    auto fh = f * h;

    auto adjA00 = ei - fh;
    auto adjA10 = fg - di;
    auto adjA20 = dh - eg;

    sleipnir::VariableMatrix adjA{{adjA00, ch - bi, bf - ce},
                                  {adjA10, ai - cg, cd - af},
                                  {adjA20, bg - ah, ae - bd}};
    auto detA = a * adjA00 + b * adjA10 + c * adjA20;
    return adjA / detA * B;
  } else if (A.Rows() == 4 && A.Cols() == 4) {
    // Compute optimal inverse instead of using Eigen's general solver
    //
    // [a  b  c  d]⁻¹
    // [e  f  g  h]
    // [i  j  k  l]
    // [m  n  o  p]
    //
    // https://www.wolframalpha.com/input?i=inverse+%7B%7Ba%2C+b%2C+c%2C+d%7D%2C+%7Be%2C+f%2C+g%2C+h%7D%2C+%7Bi%2C+j%2C+k%2C+l%7D%2C+%7Bm%2C+n%2C+o%2C+p%7D%7D

    const auto& a = A[0, 0];
    const auto& b = A[0, 1];
    const auto& c = A[0, 2];
    const auto& d = A[0, 3];
    const auto& e = A[1, 0];
    const auto& f = A[1, 1];
    const auto& g = A[1, 2];
    const auto& h = A[1, 3];
    const auto& i = A[2, 0];
    const auto& j = A[2, 1];
    const auto& k = A[2, 2];
    const auto& l = A[2, 3];
    const auto& m = A[3, 0];
    const auto& n = A[3, 1];
    const auto& o = A[3, 2];
    const auto& p = A[3, 3];

    auto afk = a * f * k;
    auto afl = a * f * l;
    auto afo = a * f * o;
    auto afp = a * f * p;
    auto agj = a * g * j;
    auto agl = a * g * l;
    auto agn = a * g * n;
    auto agp = a * g * p;
    auto ahj = a * h * j;
    auto ahk = a * h * k;
    auto ahn = a * h * n;
    auto aho = a * h * o;
    auto ajo = a * j * o;
    auto ajp = a * j * p;
    auto akn = a * k * n;
    auto akp = a * k * p;
    auto aln = a * l * n;
    auto alo = a * l * o;
    auto bek = b * e * k;
    auto bel = b * e * l;
    auto beo = b * e * o;
    auto bep = b * e * p;
    auto bgi = b * g * i;
    auto bgl = b * g * l;
    auto bgm = b * g * m;
    auto bgp = b * g * p;
    auto bhi = b * h * i;
    auto bhk = b * h * k;
    auto bhm = b * h * m;
    auto bho = b * h * o;
    auto bio = b * i * o;
    auto bip = b * i * p;
    auto bjp = b * j * p;
    auto bkm = b * k * m;
    auto bkp = b * k * p;
    auto blm = b * l * m;
    auto blo = b * l * o;
    auto cej = c * e * j;
    auto cel = c * e * l;
    auto cen = c * e * n;
    auto cep = c * e * p;
    auto cfi = c * f * i;
    auto cfl = c * f * l;
    auto cfm = c * f * m;
    auto cfp = c * f * p;
    auto chi = c * h * i;
    auto chj = c * h * j;
    auto chm = c * h * m;
    auto chn = c * h * n;
    auto cin = c * i * n;
    auto cip = c * i * p;
    auto cjm = c * j * m;
    auto cjp = c * j * p;
    auto clm = c * l * m;
    auto cln = c * l * n;
    auto dej = d * e * j;
    auto dek = d * e * k;
    auto den = d * e * n;
    auto deo = d * e * o;
    auto dfi = d * f * i;
    auto dfk = d * f * k;
    auto dfm = d * f * m;
    auto dfo = d * f * o;
    auto dgi = d * g * i;
    auto dgj = d * g * j;
    auto dgm = d * g * m;
    auto dgn = d * g * n;
    auto din = d * i * n;
    auto dio = d * i * o;
    auto djm = d * j * m;
    auto djo = d * j * o;
    auto dkm = d * k * m;
    auto dkn = d * k * n;
    auto ejo = e * j * o;
    auto ejp = e * j * p;
    auto ekn = e * k * n;
    auto ekp = e * k * p;
    auto eln = e * l * n;
    auto elo = e * l * o;
    auto fio = f * i * o;
    auto fip = f * i * p;
    auto fkm = f * k * m;
    auto fkp = f * k * p;
    auto flm = f * l * m;
    auto flo = f * l * o;
    auto gin = g * i * n;
    auto gip = g * i * p;
    auto gjm = g * j * m;
    auto gjp = g * j * p;
    auto glm = g * l * m;
    auto gln = g * l * n;
    auto hin = h * i * n;
    auto hio = h * i * o;
    auto hjm = h * j * m;
    auto hjo = h * j * o;
    auto hkm = h * k * m;
    auto hkn = h * k * n;

    auto adjA00 = fkp - flo - gjp + gln + hjo - hkn;
    auto adjA01 = -bkp + blo + cjp - cln - djo + dkn;
    auto adjA02 = bgp - bho - cfp + chn + dfo - dgn;
    auto adjA03 = -bgl + bhk + cfl - chj - dfk + dgj;
    auto adjA10 = -ekp + elo + gip - glm - hio + hkm;
    auto adjA11 = akp - alo - cip + clm + dio - dkm;
    auto adjA12 = -agp + aho + cep - chm - deo + dgm;
    auto adjA13 = agl - ahk - cel + chi + dek - dgi;
    auto adjA20 = ejp - eln - fip + flm + hin - hjm;
    auto adjA21 = -ajp + aln + bip - blm - din + djm;
    auto adjA22 = afp - ahn - bep + bhm + den - dfm;
    auto adjA23 = -afl + ahj + bel - bhi - dej + dfi;
    auto adjA30 = -ejo + ekn + fio - fkm - gin + gjm;
    // NOLINTNEXTLINE
    auto adjA31 = ajo - akn - bio + bkm + cin - cjm;
    auto adjA32 = -afo + agn + beo - bgm - cen + cfm;
    auto adjA33 = afk - agj - bek + bgi + cej - cfi;

    sleipnir::VariableMatrix adjA{{adjA00, adjA01, adjA02, adjA03},
                                  {adjA10, adjA11, adjA12, adjA13},
                                  {adjA20, adjA21, adjA22, adjA23},
                                  {adjA30, adjA31, adjA32, adjA33}};
    auto detA = a * adjA00 + b * adjA10 + c * adjA20 + d * adjA30;
    return adjA / detA * B;
  } else {
    using MatrixXv = Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic>;

    MatrixXv eigenA{A.Rows(), A.Cols()};
    for (int row = 0; row < A.Rows(); ++row) {
      for (int col = 0; col < A.Cols(); ++col) {
        eigenA(row, col) = A[row, col];
      }
    }

    MatrixXv eigenB{B.Rows(), B.Cols()};
    for (int row = 0; row < B.Rows(); ++row) {
      for (int col = 0; col < B.Cols(); ++col) {
        eigenB(row, col) = B[row, col];
      }
    }

    MatrixXv eigenX = eigenA.householderQr().solve(eigenB);

    VariableMatrix X{A.Cols(), B.Cols()};
    for (int row = 0; row < X.Rows(); ++row) {
      for (int col = 0; col < X.Cols(); ++col) {
        X[row, col] = eigenX(row, col);
      }
    }

    return X;
  }
}

}  // namespace sleipnir
