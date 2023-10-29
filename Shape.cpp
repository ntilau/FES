#include "Shape.h"

Shape::Shape(size_t pOrd, size_t cDim, STYPE sType, arma::rowvec cPos, Jacobian* cJac)
{
    switch(cDim)
    {
    case 3: // TETRAHEDRON
        switch(sType)
        {
        case Hcurl:
            Ns.resize(1,4);
            dNs.resize(3,4);
            Ns(0) = 1-cPos(0)-cPos(1)-cPos(2);
            Ns(1) = cPos(0);
            Ns(2) = cPos(1);
            Ns(3) = cPos(2);
            dNs(0,0) = -1;
            dNs(1,0) = -1;
            dNs(2,0) = -1;
            dNs(0,1) = 1;
            dNs(1,2) = 1;
            dNs(2,3) = 1;
            dNs = cJac->invJ * dNs;
            switch(pOrd)
            {
            case 1:
                Nv.resize(3,6);
                dNv.resize(3,6);
                Nv.col(0) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
                Nv.col(1) = Ns(0)*dNs.col(2)-Ns(2)*dNs.col(0);
                Nv.col(2) = Ns(0)*dNs.col(3)-Ns(3)*dNs.col(0);
                Nv.col(3) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
                Nv.col(4) = Ns(1)*dNs.col(3)-Ns(3)*dNs.col(1);
                Nv.col(5) = Ns(2)*dNs.col(3)-Ns(3)*dNs.col(2);
                dNv.col(0) = 2*cross(dNs.col(0),dNs.col(1));
                dNv.col(1) = 2*cross(dNs.col(0),dNs.col(2));
                dNv.col(2) = 2*cross(dNs.col(0),dNs.col(3));
                dNv.col(3) = 2*cross(dNs.col(1),dNs.col(2));
                dNv.col(4) = 2*cross(dNs.col(1),dNs.col(3));
                dNv.col(5) = 2*cross(dNs.col(2),dNs.col(3));
                break;
            case 2:
                Nv.resize(3,20);
                dNv.resize(3,20);
                Nv.col(0) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
                Nv.col(1) = Ns(0)*dNs.col(2)-Ns(2)*dNs.col(0);
                Nv.col(2) = Ns(0)*dNs.col(3)-Ns(3)*dNs.col(0);
                Nv.col(3) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
                Nv.col(4) = Ns(1)*dNs.col(3)-Ns(3)*dNs.col(1);
                Nv.col(5) = Ns(2)*dNs.col(3)-Ns(3)*dNs.col(2);
                // 2nd ord
                Nv.col(6) = 4*Ns(0)*dNs.col(1)+4*Ns(1)*dNs.col(0);
                Nv.col(7) = 4*Ns(0)*dNs.col(2)+4*Ns(2)*dNs.col(0);
                Nv.col(8) = 4*Ns(0)*dNs.col(3)+4*Ns(3)*dNs.col(0);
                Nv.col(9) = 4*Ns(1)*dNs.col(2)+4*Ns(2)*dNs.col(1);
                Nv.col(10) = 4*Ns(1)*dNs.col(3)+4*Ns(3)*dNs.col(1);
                Nv.col(11) = 4*Ns(2)*dNs.col(3)+4*Ns(3)*dNs.col(2);
                Nv.col(12) = Ns(1)*Ns(2)*dNs.col(3)-Ns(1)*Ns(3)*dNs.col(2);
                Nv.col(13) = Ns(1)*Ns(2)*dNs.col(3)-Ns(2)*Ns(3)*dNs.col(1);
                Nv.col(14) = Ns(0)*Ns(2)*dNs.col(3)-Ns(0)*Ns(3)*dNs.col(2);
                Nv.col(15) = Ns(0)*Ns(2)*dNs.col(3)-Ns(2)*Ns(3)*dNs.col(0);
                Nv.col(16) = Ns(0)*Ns(1)*dNs.col(3)-Ns(0)*Ns(3)*dNs.col(1);
                Nv.col(17) = Ns(0)*Ns(1)*dNs.col(3)-Ns(1)*Ns(3)*dNs.col(0);
                Nv.col(18) = Ns(0)*Ns(1)*dNs.col(2)-Ns(0)*Ns(2)*dNs.col(1);
                Nv.col(19) = Ns(0)*Ns(1)*dNs.col(2)-Ns(1)*Ns(2)*dNs.col(0);
                dNv.col(0) = 2*cross(dNs.col(0),dNs.col(1));
                dNv.col(1) = 2*cross(dNs.col(0),dNs.col(2));
                dNv.col(2) = 2*cross(dNs.col(0),dNs.col(3));
                dNv.col(3) = 2*cross(dNs.col(1),dNs.col(2));
                dNv.col(4) = 2*cross(dNs.col(1),dNs.col(3));
                dNv.col(5) = 2*cross(dNs.col(2),dNs.col(3));
                dNv.col(12) = 2*Ns(1)*cross(dNs.col(2),dNs.col(3)) +
                              Ns(2)*cross(dNs.col(1),dNs.col(3)) -
                              Ns(3)*cross(dNs.col(1),dNs.col(2));
                dNv.col(13) = Ns(1)*cross(dNs.col(2),dNs.col(3)) +
                              2*Ns(2)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(3)*cross(dNs.col(1),dNs.col(2));
                dNv.col(14) = 2*Ns(0)*cross(dNs.col(2),dNs.col(3)) +
                              Ns(2)*cross(dNs.col(0),dNs.col(3)) -
                              Ns(3)*cross(dNs.col(0),dNs.col(2));
                dNv.col(15) = Ns(0)*cross(dNs.col(2),dNs.col(3)) +
                              2*Ns(2)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(3)*cross(dNs.col(0),dNs.col(2));
                dNv.col(16) = 2*Ns(0)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(1)*cross(dNs.col(0),dNs.col(3)) -
                              Ns(3)*cross(dNs.col(0),dNs.col(1));
                dNv.col(17) = Ns(0)*cross(dNs.col(1),dNs.col(3)) +
                              2*Ns(1)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(3)*cross(dNs.col(0),dNs.col(1));
                dNv.col(18) = 2*Ns(0)*cross(dNs.col(1),dNs.col(2)) +
                              Ns(1)*cross(dNs.col(0),dNs.col(2)) -
                              Ns(2)*cross(dNs.col(0),dNs.col(1));
                dNv.col(19) = Ns(0)*cross(dNs.col(1),dNs.col(2)) +
                              2*Ns(1)*cross(dNs.col(0),dNs.col(2)) +
                              Ns(2)*cross(dNs.col(0),dNs.col(1));
                break;
            case 3:
                Nv.resize(3,45);
                dNv.resize(3,45);
                Nv.col(0) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
                Nv.col(1) = Ns(0)*dNs.col(2)-Ns(2)*dNs.col(0);
                Nv.col(2) = Ns(0)*dNs.col(3)-Ns(3)*dNs.col(0);
                Nv.col(3) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
                Nv.col(4) = Ns(1)*dNs.col(3)-Ns(3)*dNs.col(1);
                Nv.col(5) = Ns(2)*dNs.col(3)-Ns(3)*dNs.col(2);
                Nv.col(6) = 4*Ns(0)*dNs.col(1)+4*Ns(1)*dNs.col(0);
                Nv.col(7) = 4*Ns(0)*dNs.col(2)+4*Ns(2)*dNs.col(0);
                Nv.col(8) = 4*Ns(0)*dNs.col(3)+4*Ns(3)*dNs.col(0);
                Nv.col(9) = 4*Ns(1)*dNs.col(2)+4*Ns(2)*dNs.col(1);
                Nv.col(10) = 4*Ns(1)*dNs.col(3)+4*Ns(3)*dNs.col(1);
                Nv.col(11) = 4*Ns(2)*dNs.col(3)+4*Ns(3)*dNs.col(2);
                Nv.col(12) = Ns(1)*Ns(2)*dNs.col(3)-Ns(1)*Ns(3)*dNs.col(2);
                Nv.col(13) = Ns(1)*Ns(2)*dNs.col(3)-Ns(2)*Ns(3)*dNs.col(1);
                Nv.col(14) = Ns(0)*Ns(2)*dNs.col(3)-Ns(0)*Ns(3)*dNs.col(2);
                Nv.col(15) = Ns(0)*Ns(2)*dNs.col(3)-Ns(2)*Ns(3)*dNs.col(0);
                Nv.col(16) = Ns(0)*Ns(1)*dNs.col(3)-Ns(0)*Ns(3)*dNs.col(1);
                Nv.col(17) = Ns(0)*Ns(1)*dNs.col(3)-Ns(1)*Ns(3)*dNs.col(0);
                Nv.col(18) = Ns(0)*Ns(1)*dNs.col(2)-Ns(0)*Ns(2)*dNs.col(1);
                Nv.col(19) = Ns(0)*Ns(1)*dNs.col(2)-Ns(1)*Ns(2)*dNs.col(0);
                // 3rd ord
                Nv.col(20) = Ns(0)*(Ns(0)-2*Ns(1))*dNs.col(1) +
                             Ns(1)*(-Ns(1)+2*Ns(0))*dNs.col(0);
                Nv.col(21) = Ns(0)*(Ns(0)-2*Ns(2))*dNs.col(2) +
                             Ns(2)*(-Ns(2)+2*Ns(0))*dNs.col(0);
                Nv.col(22) = Ns(0)*(Ns(0)-2*Ns(3))*dNs.col(3) +
                             Ns(3)*(-Ns(3)+2*Ns(0))*dNs.col(0);
                Nv.col(23) = Ns(1)*(Ns(1)-2*Ns(2))*dNs.col(2) +
                             Ns(2)*(-Ns(2)+2*Ns(1))*dNs.col(1);
                Nv.col(24) = Ns(1)*(Ns(1)-2*Ns(3))*dNs.col(3) +
                             Ns(3)*(-Ns(3)+2*Ns(1))*dNs.col(1);
                Nv.col(25) = Ns(2)*(Ns(2)-2*Ns(3))*dNs.col(3) +
                             Ns(3)*(-Ns(3)+2*Ns(2))*dNs.col(2);
                Nv.col(26) = Ns(1)*Ns(2)*dNs.col(3) +
                             Ns(1)*Ns(3)*dNs.col(2) +
                             Ns(2)*Ns(3)*dNs.col(1);
                Nv.col(27) = -Ns(1)*Ns(2)*(Ns(2)-2*Ns(3))*dNs.col(3) -
                             Ns(1)*Ns(3)*(-Ns(3)+2*Ns(2))*dNs.col(2) +
                             3*Ns(2)*Ns(3)*(Ns(2)-Ns(3))*dNs.col(1);
                Nv.col(28) = -Ns(1)*Ns(2)*(Ns(1)-2*Ns(3))*dNs.col(3) -
                             Ns(2)*Ns(3)*(-Ns(3)+2*Ns(1))*dNs.col(1) +
                             3*Ns(1)*Ns(3)*(Ns(1)-Ns(3))*dNs.col(2);
                Nv.col(29) = -Ns(1)*Ns(3)*(Ns(1)-2*Ns(2))*dNs.col(2) -
                             Ns(2)*Ns(3)*(-Ns(2)+2*Ns(1))*dNs.col(1) +
                             3*Ns(1)*Ns(2)*(Ns(1)-Ns(2))*dNs.col(3);
                Nv.col(30) = Ns(0)*Ns(2)*dNs.col(3) +
                             Ns(0)*Ns(3)*dNs.col(2) +
                             Ns(2)*Ns(3)*dNs.col(0);
                Nv.col(31) = -Ns(0)*Ns(2)*(Ns(2)-2*Ns(3))*dNs.col(3) -
                             Ns(0)*Ns(3)*(-Ns(3)+2*Ns(2))*dNs.col(2) +
                             3*Ns(2)*Ns(3)*(Ns(2)-Ns(3))*dNs.col(0);
                Nv.col(32) = -Ns(0)*Ns(2)*(Ns(0)-2*Ns(3))*dNs.col(3) -
                             Ns(2)*Ns(3)*(-Ns(3)+2*Ns(0))*dNs.col(0) +
                             3*Ns(0)*Ns(3)*(Ns(0)-Ns(3))*dNs.col(2);
                Nv.col(33) = -Ns(0)*Ns(3)*(Ns(0)-2*Ns(2))*dNs.col(2) -
                             Ns(2)*Ns(3)*(-Ns(2)+2*Ns(0))*dNs.col(0) +
                             3*Ns(0)*Ns(2)*(Ns(0)-Ns(2))*dNs.col(3);
                Nv.col(34) = Ns(0)*Ns(1)*dNs.col(3) +
                             Ns(0)*Ns(3)*dNs.col(1) +
                             Ns(1)*Ns(3)*dNs.col(0);
                Nv.col(35) = -Ns(0)*Ns(1)*(Ns(1)-2*Ns(3))*dNs.col(3) -
                             Ns(0)*Ns(3)*(-Ns(3)+2*Ns(1))*dNs.col(1) +
                             3*Ns(1)*Ns(3)*(Ns(1)-Ns(3))*dNs.col(0);
                Nv.col(36) = -Ns(0)*Ns(1)*(Ns(0)-2*Ns(3))*dNs.col(3) -
                             Ns(1)*Ns(3)*(-Ns(3)+2*Ns(0))*dNs.col(0) +
                             3*Ns(0)*Ns(3)*(Ns(0)-Ns(3))*dNs.col(1);
                Nv.col(37) = -Ns(0)*Ns(3)*(Ns(0)-2*Ns(1))*dNs.col(1) -
                             Ns(1)*Ns(3)*(-Ns(1)+2*Ns(0))*dNs.col(0) +
                             3*Ns(0)*Ns(1)*(Ns(0)-Ns(1))*dNs.col(3);
                Nv.col(38) = Ns(0)*Ns(1)*dNs.col(2) +
                             Ns(0)*Ns(2)*dNs.col(1) +
                             Ns(1)*Ns(2)*dNs.col(0);
                Nv.col(39) = -Ns(0)*Ns(1)*(Ns(1)-2*Ns(2))*dNs.col(2) -
                             Ns(0)*Ns(2)*(-Ns(2)+2*Ns(1))*dNs.col(1) +
                             3*Ns(1)*Ns(2)*(Ns(1)-Ns(2))*dNs.col(0);
                Nv.col(40) = -Ns(0)*Ns(1)*(Ns(0)-2*Ns(2))*dNs.col(2) -
                             Ns(1)*Ns(2)*(-Ns(2)+2*Ns(0))*dNs.col(0) +
                             3*Ns(0)*Ns(2)*(Ns(0)-Ns(2))*dNs.col(1);
                Nv.col(41) = -Ns(0)*Ns(2)*(Ns(0)-2*Ns(1))*dNs.col(1) -
                             Ns(1)*Ns(2)*(-Ns(1)+2*Ns(0))*dNs.col(0) +
                             3*Ns(0)*Ns(1)*(Ns(0)-Ns(1))*dNs.col(2);
                Nv.col(42) = -Ns(0)*Ns(1)*Ns(2)*dNs.col(3) -
                             Ns(1)*Ns(0)*Ns(3)*dNs.col(2) -
                             Ns(0)*Ns(2)*Ns(3)*dNs.col(1) +
                             3*Ns(1)*Ns(2)*Ns(3)*dNs.col(0);
                Nv.col(43) = -Ns(0)*Ns(1)*Ns(2)*dNs.col(3) -
                             Ns(1)*Ns(0)*Ns(3)*dNs.col(2) -
                             Ns(1)*Ns(2)*Ns(3)*dNs.col(0) +
                             3*Ns(0)*Ns(2)*Ns(3)*dNs.col(1);
                Nv.col(44) = -Ns(0)*Ns(1)*Ns(2)*dNs.col(3) -
                             Ns(0)*Ns(2)*Ns(3)*dNs.col(1) -
                             Ns(1)*Ns(2)*Ns(3)*dNs.col(0) +
                             3*Ns(1)*Ns(0)*Ns(3)*dNs.col(2);
                dNv.col(0) = 2*cross(dNs.col(0),dNs.col(1));
                dNv.col(1) = 2*cross(dNs.col(0),dNs.col(2));
                dNv.col(2) = 2*cross(dNs.col(0),dNs.col(3));
                dNv.col(3) = 2*cross(dNs.col(1),dNs.col(2));
                dNv.col(4) = 2*cross(dNs.col(1),dNs.col(3));
                dNv.col(5) = 2*cross(dNs.col(2),dNs.col(3));
                dNv.col(12) = 2*Ns(1)*cross(dNs.col(2),dNs.col(3)) +
                              Ns(2)*cross(dNs.col(1),dNs.col(3)) -
                              Ns(3)*cross(dNs.col(1),dNs.col(2));
                dNv.col(13) = Ns(1)*cross(dNs.col(2),dNs.col(3)) +
                              2*Ns(2)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(3)*cross(dNs.col(1),dNs.col(2));
                dNv.col(14) = 2*Ns(0)*cross(dNs.col(2),dNs.col(3)) +
                              Ns(2)*cross(dNs.col(0),dNs.col(3)) -
                              Ns(3)*cross(dNs.col(0),dNs.col(2));
                dNv.col(15) = Ns(0)*cross(dNs.col(2),dNs.col(3)) +
                              2*Ns(2)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(3)*cross(dNs.col(0),dNs.col(2));
                dNv.col(16) = 2*Ns(0)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(1)*cross(dNs.col(0),dNs.col(3)) -
                              Ns(3)*cross(dNs.col(0),dNs.col(1));
                dNv.col(17) = Ns(0)*cross(dNs.col(1),dNs.col(3)) +
                              2*Ns(1)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(3)*cross(dNs.col(0),dNs.col(1));
                dNv.col(18) = 2*Ns(0)*cross(dNs.col(1),dNs.col(2)) +
                              Ns(1)*cross(dNs.col(0),dNs.col(2)) -
                              Ns(2)*cross(dNs.col(0),dNs.col(1));
                dNv.col(19) = Ns(0)*cross(dNs.col(1),dNs.col(2)) +
                              2*Ns(1)*cross(dNs.col(0),dNs.col(2)) +
                              Ns(2)*cross(dNs.col(0),dNs.col(1));
                dNv.col(27) = -4*Ns(2)*(Ns(2)-2*Ns(3))*cross(dNs.col(1),dNs.col(3)) -
                              4*Ns(3)*(2*Ns(2)-Ns(3))*cross(dNs.col(1),dNs.col(2));
                dNv.col(28) = -4*Ns(1)*(Ns(1)-2*Ns(3))*cross(dNs.col(2),dNs.col(3)) +
                              4*Ns(3)*(2*Ns(1)-Ns(3))*cross(dNs.col(1),dNs.col(2));
                dNv.col(29) = 4*Ns(1)*(Ns(1)-2*Ns(2))*cross(dNs.col(2),dNs.col(3)) +
                              4*Ns(2)*(2*Ns(1)-Ns(2))*cross(dNs.col(1),dNs.col(3));
                dNv.col(31) = -4*Ns(2)*(Ns(2)-2*Ns(3))*cross(dNs.col(0),dNs.col(3)) -
                              4*Ns(3)*(2*Ns(2)-Ns(3))*cross(dNs.col(0),dNs.col(2));
                dNv.col(32) = -4*Ns(0)*(Ns(0)-2*Ns(3))*cross(dNs.col(2),dNs.col(3)) +
                              4*Ns(3)*(2*Ns(0)-Ns(3))*cross(dNs.col(0),dNs.col(2));
                dNv.col(33) = 4*Ns(0)*(Ns(0)-2*Ns(2))*cross(dNs.col(2),dNs.col(3)) +
                              4*Ns(2)*(2*Ns(0)-Ns(2))*cross(dNs.col(0),dNs.col(3));
                dNv.col(35) = -4*Ns(1)*(Ns(1)-2*Ns(3))*cross(dNs.col(0),dNs.col(3)) -
                              4*Ns(3)*(2*Ns(1)-Ns(3))*cross(dNs.col(0),dNs.col(1));
                dNv.col(36) = -4*Ns(0)*(Ns(0)-2*Ns(3))*cross(dNs.col(1),dNs.col(3)) +
                              4*Ns(3)*(2*Ns(0)-Ns(3))*cross(dNs.col(0),dNs.col(1));
                dNv.col(37) = 4*Ns(0)*(Ns(0)-2*Ns(1))*cross(dNs.col(1),dNs.col(3)) +
                              4*Ns(1)*(2*Ns(0)-Ns(1))*cross(dNs.col(0),dNs.col(3));
                dNv.col(39) = -4*Ns(1)*(Ns(1)-2*Ns(2))*cross(dNs.col(0),dNs.col(2)) -
                              4*Ns(2)*(2*Ns(1)-Ns(2))*cross(dNs.col(0),dNs.col(1));
                dNv.col(40) = -4*Ns(0)*(Ns(0)-2*Ns(2))*cross(dNs.col(1),dNs.col(2)) +
                              4*Ns(2)*(2*Ns(0)-Ns(2))*cross(dNs.col(0),dNs.col(1));
                dNv.col(41) = 4*Ns(0)*(Ns(0)-2*Ns(1))*cross(dNs.col(1),dNs.col(2)) +
                              4*Ns(1)*(2*Ns(0)-Ns(1))*cross(dNs.col(0),dNs.col(2));
                dNv.col(42) = -4*Ns(1)*Ns(2)*cross(dNs.col(0),dNs.col(3)) -
                              4*Ns(1)*Ns(3)*cross(dNs.col(0),dNs.col(2)) -
                              4*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(1));
                dNv.col(43) = -4*Ns(0)*Ns(2)*cross(dNs.col(1),dNs.col(3)) -
                              4*Ns(0)*Ns(3)*cross(dNs.col(1),dNs.col(2)) +
                              4*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(1));
                dNv.col(44) = -4*Ns(1)*Ns(0)*cross(dNs.col(2),dNs.col(3)) +
                              4*Ns(0)*Ns(3)*cross(dNs.col(1),dNs.col(2)) +
                              4*Ns(1)*Ns(3)*cross(dNs.col(0),dNs.col(2));
                break;
            default:
                throw std::string("ShapeHcurlTetra order not yet implemented");
            }
            break;
        case Hgrad:
            switch(pOrd)
            {
            case 1:
                Ns.resize(1,4);
                dNs.resize(3,4);
                Ns(0) = 1-cPos(0)-cPos(1)-cPos(2);
                Ns(1) = cPos(0);
                Ns(2) = cPos(1);
                Ns(3) = cPos(2);
                dNs(0,0) = -1;
                dNs(1,0) = -1;
                dNs(2,0) = -1;
                dNs(0,1) = 1;
                dNs(1,2) = 1;
                dNs(2,3) = 1;
                dNs = cJac->invJ * dNs;
                break;
            case 2:
                // hierarchical basis functions are not appropriate for Hgrad space
                Ns.resize(1,10);
                dNs.resize(3,10);
                dNs.fill(0);
                cPos.resize(4);
                cPos(3) = 1-cPos(0)-cPos(1)-cPos(2);
                Ns(0) = cPos(3)*(2*cPos(3) - 1);
                Ns(1) = cPos(0)*(2*cPos(0) - 1);
                Ns(2) = cPos(1)*(2*cPos(1) - 1);
                Ns(3) = cPos(2)*(2*cPos(2) - 1);
                Ns(4) = cPos(3)*cPos(0)*4;
                Ns(5) = cPos(3)*cPos(1)*4;
                Ns(6) = cPos(3)*cPos(2)*4;
                Ns(7) = cPos(0)*cPos(1)*4;
                Ns(8) = cPos(0)*cPos(2)*4;
                Ns(9) = cPos(1)*cPos(2)*4;
                dNs(0,0) = 4*cPos(0) + 4*cPos(1) + 4*cPos(2) - 3;
                dNs(1,0) = 4*cPos(0) + 4*cPos(1) + 4*cPos(2) - 3;
                dNs(2,0) = 4*cPos(0) + 4*cPos(1) + 4*cPos(2) - 3;
                dNs(0,1) = 4*cPos(0) - 1;
                dNs(1,2) = 4*cPos(1) - 1;
                dNs(2,3) = 4*cPos(2) - 1;
                dNs(0,4) = (cPos(3)-cPos(0))*4;
                dNs(1,4) = -cPos(0)*4;
                dNs(2,4) = -cPos(0)*4;
                dNs(0,5) = -cPos(1)*4;
                dNs(1,5) = (cPos(3)-cPos(1))*4;
                dNs(2,5) = -cPos(1)*4;
                dNs(0,6) = -cPos(2)*4;
                dNs(1,6) = -cPos(2)*4;
                dNs(2,6) = (cPos(3)-cPos(2))*4;
                dNs(0,7) = cPos(1)*4;
                dNs(1,7) = cPos(0)*4;
                dNs(0,8) = cPos(2)*4;
                dNs(2,8) = cPos(0)*4;
                dNs(1,9) = cPos(2)*4;
                dNs(2,9) = cPos(1)*4;
                dNs = cJac->invJ * dNs;
                break;
            case 3:
                Ns.resize(1,20);
                dNs.resize(3,20);
                //dNs.fill(0);
                cPos.resize(4);
                cPos(3) = 1-cPos(0)-cPos(1)-cPos(2);
                Ns(0) = (cPos(3)*(3*cPos(3) - 1)*(3*cPos(3) - 2))/2;
                Ns(1) = (cPos(0)*(3*cPos(0) - 1)*(3*cPos(0) - 2))/2;
                Ns(2) = (cPos(1)*(3*cPos(1) - 1)*(3*cPos(1) - 2))/2;
                Ns(3) = (cPos(2)*(3*cPos(2) - 1)*(3*cPos(2) - 2))/2;
                Ns(4) = (9*cPos(3)*cPos(0)*(3*cPos(3) - 1))/2;
                Ns(10) = (9*cPos(3)*cPos(0)*(3*cPos(0) - 1))/2;
                Ns(5) = (9*cPos(3)*cPos(1)*(3*cPos(3) - 1))/2;
                Ns(11) = (9*cPos(3)*cPos(1)*(3*cPos(1) - 1))/2;
                Ns(6) = (9*cPos(3)*cPos(2)*(3*cPos(3) - 1))/2;
                Ns(12) = (9*cPos(3)*cPos(2)*(3*cPos(2) - 1))/2;
                Ns(7) = (9*cPos(0)*cPos(1)*(3*cPos(0) - 1))/2;
                Ns(13) = (9*cPos(0)*cPos(1)*(3*cPos(1) - 1))/2;
                Ns(8) = (9*cPos(0)*cPos(2)*(3*cPos(0) - 1))/2;
                Ns(14) = (9*cPos(0)*cPos(2)*(3*cPos(2) - 1))/2;
                Ns(9) = (9*cPos(1)*cPos(2)*(3*cPos(1) - 1))/2;
                Ns(15) = (9*cPos(1)*cPos(2)*(3*cPos(2) - 1))/2;
                Ns(16) = 27*cPos(0)*cPos(1)*cPos(2);
                Ns(17) = 27*cPos(3)*cPos(1)*cPos(2);
                Ns(18) = 27*cPos(3)*cPos(0)*cPos(2);
                Ns(19) = 27*cPos(3)*cPos(0)*cPos(1);
                dNs(0,0) =  18*cPos(0) + 18*cPos(1) + 18*cPos(2) - 27*cPos(0)*cPos(1) - 27*cPos(0)*cPos(2) - 27*cPos(1)*cPos(2) -
                            (27*pow(cPos(0),2))/2 - (27*pow(cPos(1),2))/2 - (27*pow(cPos(2),2))/2 - 11.0/2.0;
                dNs(1,0) =  18*cPos(0) + 18*cPos(1) + 18*cPos(2) - 27*cPos(0)*cPos(1) - 27*cPos(0)*cPos(2) - 27*cPos(1)*cPos(2) -
                            (27*pow(cPos(0),2))/2 - (27*pow(cPos(1),2))/2 - (27*pow(cPos(2),2))/2 - 11.0/2.0;
                dNs(2,0) =  18*cPos(0) + 18*cPos(1) + 18*cPos(2) - 27*cPos(0)*cPos(1) - 27*cPos(0)*cPos(2) - 27*cPos(1)*cPos(2) -
                            (27*pow(cPos(0),2))/2 - (27*pow(cPos(1),2))/2 - (27*pow(cPos(2),2))/2 - 11.0/2.0;
                dNs(0,1) = (27*pow(cPos(0),2))/2 - 9*cPos(0) + 1;
                dNs(1,2) = (27*pow(cPos(1),2))/2 - 9*cPos(1) + 1;
                dNs(2,3) = (27*pow(cPos(2),2))/2 - 9*cPos(2) + 1;
                dNs(0,4) = (81*pow(cPos(0),2))/2 + 54*cPos(0)*cPos(1) + 54*cPos(0)*cPos(2) - 45*cPos(0) + (27*pow(cPos(1),2))/2 +
                           27*cPos(1)*cPos(2) - (45*cPos(1))/2 + (27*pow(cPos(2),2))/2 - (45*cPos(2))/2 + 9;
                dNs(1,4) = (9*cPos(0)*(6*cPos(0) + 6*cPos(1) + 6*cPos(2) - 5))/2;
                dNs(2,4) = (9*cPos(0)*(6*cPos(0) + 6*cPos(1) + 6*cPos(2) - 5))/2;
                dNs(0,10) =  36*cPos(0) + (9*cPos(1))/2 + (9*cPos(2))/2 - 27*cPos(0)*cPos(1) - 27*cPos(0)*cPos(2) -
                             (81*pow(cPos(0),2))/2 - 9.0/2.0;
                dNs(1,10) = -(9*cPos(0)*(3*cPos(0) - 1))/2;
                dNs(2,10) = -(9*cPos(0)*(3*cPos(0) - 1))/2;
                dNs(0,5) = (9*cPos(1)*(6*cPos(0) + 6*cPos(1) + 6*cPos(2) - 5))/2;
                dNs(1,5) = (27*pow(cPos(0),2))/2 + 54*cPos(0)*cPos(1) + 27*cPos(0)*cPos(2) - (45*cPos(0))/2 + (81*pow(cPos(1),2))/2 +
                           54*cPos(1)*cPos(2) - 45*cPos(1) + (27*pow(cPos(2),2))/2 - (45*cPos(2))/2 + 9;
                dNs(2,5) = (9*cPos(1)*(6*cPos(0) + 6*cPos(1) + 6*cPos(2) - 5))/2;
                dNs(0,11) = -(9*cPos(1)*(3*cPos(1) - 1))/2;
                dNs(1,11) = (9*cPos(0))/2 + 36*cPos(1) + (9*cPos(2))/2 - 27*cPos(0)*cPos(1) - 27*cPos(1)*cPos(2) -
                            (81*pow(cPos(1),2))/2 - 9.0/2.0;
                dNs(2,11) = -(9*cPos(1)*(3*cPos(1) - 1))/2;
                dNs(0,6) = (9*cPos(2)*(6*cPos(0) + 6*cPos(1) + 6*cPos(2) - 5))/2;
                dNs(1,6) = (9*cPos(2)*(6*cPos(0) + 6*cPos(1) + 6*cPos(2) - 5))/2;
                dNs(2,6) = (27*pow(cPos(0),2))/2 + 27*cPos(0)*cPos(1) + 54*cPos(0)*cPos(2) - (45*cPos(0))/2 + (27*pow(cPos(1),2))/2 +
                           54*cPos(1)*cPos(2) - (45*cPos(1))/2 + (81*pow(cPos(2),2))/2 - 45*cPos(2) + 9;
                dNs(0,12) = -(9*cPos(2)*(3*cPos(2) - 1))/2;
                dNs(1,12) = -(9*cPos(2)*(3*cPos(2) - 1))/2;
                dNs(2,12) = (9*cPos(0))/2 + (9*cPos(1))/2 + 36*cPos(2) - 27*cPos(0)*cPos(2) - 27*cPos(1)*cPos(2) -
                            (81*pow(cPos(2),2))/2 - 9.0/2.0;
                dNs(0,7) = (9*cPos(1)*(6*cPos(0) - 1))/2;
                dNs(1,7) = (9*cPos(0)*(3*cPos(0) - 1))/2;
                dNs(0,13) = (9*cPos(1)*(3*cPos(1) - 1))/2;
                dNs(1,13) = (9*cPos(0)*(6*cPos(1) - 1))/2;
                dNs(0,8) = (9*cPos(2)*(6*cPos(0) - 1))/2;
                dNs(2,8) = (9*cPos(0)*(3*cPos(0) - 1))/2;
                dNs(0,14) = (9*cPos(2)*(3*cPos(2) - 1))/2;
                dNs(2,14) = (9*cPos(0)*(6*cPos(2) - 1))/2;
                dNs(1,9) = (9*cPos(2)*(6*cPos(1) - 1))/2;
                dNs(2,9) = (9*cPos(1)*(3*cPos(1) - 1))/2;
                dNs(1,15) = (9*cPos(2)*(3*cPos(2) - 1))/2;
                dNs(2,15) = (9*cPos(1)*(6*cPos(2) - 1))/2;
                dNs(0,16) = 27*cPos(1)*cPos(2);
                dNs(1,16) = 27*cPos(0)*cPos(2);
                dNs(2,16) = 27*cPos(0)*cPos(1);
                dNs(0,17) = -27*cPos(1)*cPos(2);
                dNs(1,17) = -27*cPos(2)*(cPos(0) + 2*cPos(1) + cPos(2) - 1);
                dNs(2,17) = -27*cPos(1)*(cPos(0) + cPos(1) + 2*cPos(2) - 1);
                dNs(0,18) = -27*cPos(2)*(2*cPos(0) + cPos(1) + cPos(2) - 1);
                dNs(1,18) = -27*cPos(0)*cPos(2);
                dNs(2,18) = -27*cPos(0)*(cPos(0) + cPos(1) + 2*cPos(2) - 1);
                dNs(0,19) = -27*cPos(1)*(2*cPos(0) + cPos(1) + cPos(2) - 1);
                dNs(1,19) = -27*cPos(0)*(cPos(0) + 2*cPos(1) + cPos(2) - 1);
                dNs(2,19) = -27*cPos(0)*cPos(1);
                dNs = cJac->invJ * dNs;
                break;
            default:
                throw std::string("ShapeHgradTetra order not yet implemented");
            }
            break;
        }
        break;
    case 2: // TRIANGLE
        Ns.resize(1,3);
        dNs.resize(2,3);
        cNs.resize(2,3);
        Ns(0) = 1-cPos(0)-cPos(1);
        Ns(1) = cPos(0);
        Ns(2) = cPos(1);
        dNs(0,0) = -1;
        dNs(1,0) = -1;
        dNs(0,1) = 1;
        dNs(1,2) = 1;
        cNs(0,0) = -1;
        cNs(1,0) = 1;
        cNs(1,1) = -1;
        cNs(0,2) = 1;
        cNs = cJac->invJ * cNs;
        dNs = cJac->invJ * dNs;
        switch(pOrd)
        {
        case 1:
            Nv.resize(2,3);
            dNv.resize(1,3);
            cNv.resize(2,3);
            Nv.col(0) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
            Nv.col(1) = Ns(0)*dNs.col(2)-Ns(2)*dNs.col(0);
            Nv.col(2) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
            cNv(0,0) = -std::sqrt(2)*Ns(1);
            cNv(1,0) = -std::sqrt(2)*Ns(2);
            cNv(0,1) = -Ns(1)+1.0;
            cNv(1,1) = Ns(2);
            cNv(0,2) = Ns(1);
            cNv(1,2) = -Ns(2)+1.0;
            dNv(0,0) = 2;
            dNv(0,1) = -2;
            dNv(0,2) = 2;
            cNv *= cNs(0,1)*cNs(1,2)-cNs(1,1)*cNs(0,2);
            dNv *= dNs(0,1)*dNs(1,2)-dNs(1,1)*dNs(0,2);
            break;
        case 2:
            Nv.resize(2,8);
            dNv.resize(1,8);
            Nv.col(0) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
            Nv.col(1) = Ns(0)*dNs.col(2)-Ns(2)*dNs.col(0);
            Nv.col(2) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
            Nv.col(3) = 4*(Ns(1)*dNs.col(2)+Ns(2)*dNs.col(1));
            Nv.col(4) = 4*(Ns(0)*dNs.col(2)+Ns(2)*dNs.col(0));
            Nv.col(5) = 4*(Ns(0)*dNs.col(1)+Ns(1)*dNs.col(0));
            Nv.col(6) = Ns(0)*Ns(1)*dNs.col(2)-Ns(0)*Ns(2)*dNs.col(1);
            Nv.col(7) = Ns(0)*Ns(1)*dNs.col(2)-Ns(1)*Ns(2)*dNs.col(0);
            dNv(0,0) = 2;
            dNv(0,1) = -2;
            dNv(0,2) = 2;
            dNv(0,6) = 2*Ns(0)-Ns(1)-Ns(2);
            dNv(0,7) = Ns(0)+Ns(2)-2*Ns(1);
            dNv *= dNs(0,1)*dNs(1,2)-dNs(1,1)*dNs(0,2);
            // update scalar functions
            Ns.resize(1,6);
            dNs.resize(2,6);
            dNs.fill(0);
            cPos.resize(3);
            cPos(2) = 1-cPos(0)-cPos(1);
            Ns(3) = 4*cPos(0)*cPos(1);
            Ns(4) = 4*cPos(2)*cPos(1);
            Ns(5) = 4*cPos(0)*cPos(2);
            dNs(0,0) = -1;
            dNs(1,0) = -1;
            dNs(0,1) = 1;
            dNs(1,2) = 1;
            dNs(0,3) = 4*cPos(1);
            dNs(0,4) = -4*cPos(1);
            dNs(0,5) = 4*(cPos(2)-cPos(0));
            dNs(1,3) = 4*cPos(0);
            dNs(1,4) = 4*(cPos(2)-cPos(1));
            dNs(1,5) = -4*cPos(0);
            dNs = cJac->invJ * dNs;
            break;
        case 3:
            Nv.resize(2,15);
            dNv.resize(1,15);
            Nv.col(0) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
            Nv.col(1) = Ns(0)*dNs.col(2)-Ns(2)*dNs.col(0);
            Nv.col(2) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
            Nv.col(3) = 4*(Ns(1)*dNs.col(2)+Ns(2)*dNs.col(1));
            Nv.col(4) = 4*(Ns(0)*dNs.col(2)+Ns(2)*dNs.col(0));
            Nv.col(5) = 4*(Ns(0)*dNs.col(1)+Ns(1)*dNs.col(0));
            Nv.col(6) = Ns(0)*Ns(1)*dNs.col(2)-Ns(0)*Ns(2)*dNs.col(1);
            Nv.col(7) = Ns(0)*Ns(1)*dNs.col(2)-Ns(1)*Ns(2)*dNs.col(0);
            Nv.col(8) = Ns(1)*(Ns(1)-2*Ns(2))*dNs.col(2) +
                        Ns(2)*(2*Ns(1)-Ns(2))*dNs.col(1);
            Nv.col(9) = Ns(0)*(Ns(0)-2*Ns(2))*dNs.col(2) +
                        Ns(2)*(2*Ns(0)-Ns(2))*dNs.col(0);
            Nv.col(10) = Ns(0)*(Ns(0)-2*Ns(1))*dNs.col(1) +
                         Ns(1)*(2*Ns(0)-Ns(1))*dNs.col(0);
            Nv.col(11) = Ns(1)*Ns(0)*dNs.col(2)+
                         Ns(0)*Ns(2)*dNs.col(1)+
                         Ns(1)*Ns(2)*dNs.col(0);
            Nv.col(12) = -Ns(1)*Ns(0)*(Ns(1)-2*Ns(2))*dNs.col(2) -
                         Ns(0)*Ns(2)*(2*Ns(1)-Ns(2))*dNs.col(1) +
                         3*Ns(1)*Ns(2)*(Ns(1)-Ns(2))*dNs.col(0);
            Nv.col(13) = -Ns(0)*Ns(1)*(Ns(0)-2*Ns(2))*dNs.col(2) -
                         Ns(1)*Ns(2)*(2*Ns(0)-Ns(2))*dNs.col(0) +
                         3*Ns(0)*Ns(2)*(Ns(0)-Ns(2))*dNs.col(1);
            Nv.col(14) = -Ns(0)*Ns(2)*(Ns(0)-2*Ns(1))*dNs.col(1) -
                         Ns(1)*Ns(2)*(2*Ns(0)-Ns(1))*dNs.col(0) +
                         3*Ns(0)*Ns(1)*(Ns(0)-Ns(1))*dNs.col(2);
            dNv(0,0) = 2;
            dNv(0,1) = -2;
            dNv(0,2) = 2;
            dNv(0,6) = 2*Ns(0)-Ns(1)-Ns(2);
            dNv(0,7) = Ns(0)+Ns(2)-2*Ns(1);
            dNv(0,12) = 4*Ns(1)*(Ns(1)-2*Ns(2)) -
                        4*Ns(2)*(2*Ns(1)-Ns(2));
            dNv(0,13) = -4*Ns(0)*(Ns(0)-2*Ns(2)) +
                        4*Ns(2)*(2*Ns(0)-Ns(2));
            dNv(0,14) = 4*Ns(0)*(Ns(0)-2*Ns(1)) -
                        4*Ns(1)*(2*Ns(0)-Ns(1));
            dNv *= dNs(0,1)*dNs(1,2)-dNs(1,1)*dNs(0,2);
            // update scalar functions
            Ns.resize(1,10);
            dNs.resize(2,10);
            dNs.fill(0);
            cPos.resize(3);
            cPos(2) = 1-cPos(0)-cPos(1);
            Ns(3) = 4*cPos(0)*cPos(1);
            Ns(4) = 4*cPos(2)*cPos(1);
            Ns(5) = 4*cPos(0)*cPos(2);
            Ns(6) = cPos(0)*cPos(1)*(cPos(0)-cPos(1));
            Ns(7) = cPos(2)*cPos(1)*(cPos(2)-cPos(1));
            Ns(8) = cPos(2)*cPos(0)*(cPos(2)-cPos(0));
            Ns(9) = cPos(2)*cPos(0)*cPos(1);
            dNs(0,0) = -1;
            dNs(1,0) = -1;
            dNs(0,1) = 1;
            dNs(1,2) = 1;
            dNs(0,3) = 4*cPos(1);
            dNs(0,4) = -4*cPos(1);
            dNs(0,5) = 4*cPos(2)-4*cPos(0);
            dNs(1,3) = 4*cPos(0);
            dNs(1,4) = 4*cPos(2)-4*cPos(1);
            dNs(1,5) = -4*cPos(0);
            dNs(0,6) = cPos(1)*(cPos(0)-cPos(1)) + cPos(0)*cPos(1);
            dNs(0,7) = -cPos(1)*(cPos(2)-cPos(1)) - cPos(2)*cPos(1);
            dNs(0,8) = cPos(2)*(cPos(2)-cPos(0)) - cPos(2)*cPos(0) - cPos(0)*(cPos(2)-cPos(0)) - cPos(2)*cPos(0);
            dNs(0,9) = (cPos(2) - cPos(0))*cPos(1);
            dNs(1,6) = cPos(0)*(cPos(0)-cPos(1)) - cPos(0)*cPos(1);
            dNs(1,7) = cPos(2)*(cPos(2)-cPos(1)) - cPos(2)*cPos(1) - cPos(1)*(cPos(2)-cPos(1)) - cPos(2)*cPos(1);
            dNs(1,8) = -cPos(0)*(cPos(2)-cPos(0)) - cPos(2)*cPos(0);
            dNs(1,9) = (cPos(2) - cPos(1))*cPos(0);
            dNs = cJac->invJ * dNs;
            break;
        default:
            throw std::string("ShapeTria order not yet implemented");
        }
        break;
    default:
        throw std::string("Shape cDim should be 2 or 3");
    }
}

Shape::~Shape()
{
    Ns.clear();
    dNs.clear();
    Nv.clear();
    dNv.clear();
}


Jacobian::Jacobian(size_t cDim, arma::mat cGeo)
{
    arma::mat dNs;
    switch(cDim)
    {
    case 3:
        dNs.resize(3,4);
        dNs.fill(0);
        dNs(0,0) = -1;
        dNs(1,0) = -1;
        dNs(2,0) = -1;
        dNs(0,1) = 1;
        dNs(1,2) = 1;
        dNs(2,3) = 1;
        break;
    case 2:
        dNs.resize(2,3);
        dNs.fill(0);
        dNs(0,0) = -1;
        dNs(1,0) = -1;
        dNs(0,1) = 1;
        dNs(1,2) = 1;
        break;
    }
    arma::mat J = dNs * cGeo;
    detJ = std::abs(arma::det(J));
    invJ = arma::inv(J);
}

Jacobian::~Jacobian()
{
    invJ.clear();
}


