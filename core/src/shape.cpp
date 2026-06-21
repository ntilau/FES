#include "shape.h"
#include <stdexcept>

shape::shape(size_t p_ord, size_t cDim, s_type sType, arma::rowvec cPos, jacobian* cJac)
{
    switch(cDim)
    {
    case 3: // TETRAHEDRON
        switch(sType)
        {
        case hcurl:
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
            switch(p_ord)
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
                dNv.fill(0);
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
            case 4:
                Nv.resize(3,84);
                dNv.resize(3,84);
                dNv.fill(0);
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
                // 4th ord edges (curl-free)
                Nv.col(45) = Ns(0)*Ns(0)*(Ns(0)-3*Ns(1))*dNs.col(1) +
                             Ns(1)*Ns(1)*(Ns(1)-3*Ns(0))*dNs.col(0);
                Nv.col(46) = Ns(0)*Ns(0)*(Ns(0)-3*Ns(2))*dNs.col(2) +
                             Ns(2)*Ns(2)*(Ns(2)-3*Ns(0))*dNs.col(0);
                Nv.col(47) = Ns(0)*Ns(0)*(Ns(0)-3*Ns(3))*dNs.col(3) +
                             Ns(3)*Ns(3)*(Ns(3)-3*Ns(0))*dNs.col(0);
                Nv.col(48) = Ns(1)*Ns(1)*(Ns(1)-3*Ns(2))*dNs.col(2) +
                             Ns(2)*Ns(2)*(Ns(2)-3*Ns(1))*dNs.col(1);
                Nv.col(49) = Ns(1)*Ns(1)*(Ns(1)-3*Ns(3))*dNs.col(3) +
                             Ns(3)*Ns(3)*(Ns(3)-3*Ns(1))*dNs.col(1);
                Nv.col(50) = Ns(2)*Ns(2)*(Ns(2)-3*Ns(3))*dNs.col(3) +
                             Ns(3)*Ns(3)*(Ns(3)-3*Ns(2))*dNs.col(2);
                // 4th ord face functions: face 0-1-2 (opposite vtx 3)
                Nv.col(51) = Ns(0)*Ns(1)*Ns(2)*dNs.col(3) +
                             Ns(1)*Ns(2)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*dNs.col(2);
                Nv.col(52) = -Ns(0)*Ns(1)*Ns(2)*(Ns(0)+Ns(1))*dNs.col(3) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(0)+Ns(1))*dNs.col(2) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(0)+Ns(2))*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(1)+Ns(2))*dNs.col(0);
                Nv.col(53) = Ns(0)*Ns(1)*Ns(2)*(Ns(2)-Ns(1))*dNs.col(3) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(2)-Ns(1))*dNs.col(2) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(0)-Ns(3))*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(0)-Ns(3))*dNs.col(0);
                Nv.col(54) = Ns(0)*Ns(1)*Ns(2)*(Ns(0)-Ns(2))*dNs.col(3) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(0)-Ns(2))*dNs.col(2) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(1)-Ns(3))*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(1)-Ns(3))*dNs.col(0);
                Nv.col(55) = Ns(0)*Ns(0)*Ns(1)*Ns(2)*dNs.col(3) +
                             Ns(1)*Ns(2)*Ns(3)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*Ns(0)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*Ns(1)*dNs.col(2);
                Nv.col(56) = Ns(0)*Ns(1)*Ns(1)*Ns(2)*dNs.col(3) +
                             Ns(1)*Ns(2)*Ns(3)*Ns(2)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*Ns(3)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*Ns(0)*dNs.col(2);
                // face 0-1-3 (opposite vtx 2): permutations with vtx 2 as opposite
                Nv.col(57) = Ns(0)*Ns(1)*Ns(3)*dNs.col(2) +
                             Ns(1)*Ns(2)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(2)*dNs.col(3);
                Nv.col(58) = -Ns(0)*Ns(1)*Ns(3)*(Ns(0)+Ns(1))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(0)+Ns(1))*dNs.col(3) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(0)+Ns(3))*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(1)+Ns(3))*dNs.col(0);
                Nv.col(59) = Ns(0)*Ns(1)*Ns(3)*(Ns(3)-Ns(1))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(3)-Ns(1))*dNs.col(3) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(0)-Ns(2))*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(0)-Ns(2))*dNs.col(0);
                Nv.col(60) = Ns(0)*Ns(1)*Ns(3)*(Ns(0)-Ns(3))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(0)-Ns(3))*dNs.col(3) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(1)-Ns(2))*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(1)-Ns(2))*dNs.col(0);
                Nv.col(61) = Ns(0)*Ns(0)*Ns(1)*Ns(3)*dNs.col(2) +
                             Ns(1)*Ns(2)*Ns(3)*Ns(2)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*Ns(0)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(2)*Ns(1)*dNs.col(3);
                Nv.col(62) = Ns(0)*Ns(1)*Ns(1)*Ns(3)*dNs.col(2) +
                             Ns(1)*Ns(2)*Ns(3)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*Ns(2)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(2)*Ns(0)*dNs.col(3);
                // face 0-2-3 (opposite vtx 1): permutations with vtx 1 as opposite
                Nv.col(63) = Ns(0)*Ns(2)*Ns(3)*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(1)*Ns(3)*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*dNs.col(3);
                Nv.col(64) = -Ns(0)*Ns(2)*Ns(3)*(Ns(0)+Ns(3))*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(0)+Ns(3))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(0)+Ns(2))*dNs.col(3) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(2)+Ns(3))*dNs.col(0);
                Nv.col(65) = Ns(0)*Ns(2)*Ns(3)*(Ns(3)-Ns(2))*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(3)-Ns(2))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(0)-Ns(1))*dNs.col(3) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(0)-Ns(1))*dNs.col(0);
                Nv.col(66) = Ns(0)*Ns(2)*Ns(3)*(Ns(0)-Ns(3))*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(0)-Ns(3))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(2)-Ns(1))*dNs.col(3) +
                             Ns(1)*Ns(2)*Ns(3)*(Ns(2)-Ns(1))*dNs.col(0);
                Nv.col(67) = Ns(0)*Ns(0)*Ns(2)*Ns(3)*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*Ns(1)*dNs.col(0) +
                             Ns(0)*Ns(1)*Ns(3)*Ns(0)*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*Ns(2)*dNs.col(3);
                Nv.col(68) = Ns(0)*Ns(2)*Ns(2)*Ns(3)*dNs.col(1) +
                             Ns(1)*Ns(2)*Ns(3)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(1)*Ns(3)*Ns(2)*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*Ns(0)*dNs.col(3);
                // face 1-2-3 (opposite vtx 0): permutations with vtx 0 as opposite
                Nv.col(69) = Ns(1)*Ns(2)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*dNs.col(3);
                Nv.col(70) = -Ns(1)*Ns(2)*Ns(3)*(Ns(2)+Ns(3))*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(2)+Ns(3))*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(1)+Ns(3))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(1)+Ns(2))*dNs.col(3);
                Nv.col(71) = Ns(1)*Ns(2)*Ns(3)*(Ns(3)-Ns(2))*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(0)-Ns(1))*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(3)-Ns(1))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(2)-Ns(1))*dNs.col(3);
                Nv.col(72) = Ns(1)*Ns(2)*Ns(3)*(Ns(1)-Ns(3))*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*(Ns(0)-Ns(2))*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*(Ns(1)-Ns(3))*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*(Ns(0)-Ns(1))*dNs.col(3);
                Nv.col(73) = Ns(1)*Ns(1)*Ns(2)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*Ns(2)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*Ns(1)*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*Ns(3)*dNs.col(3);
                Nv.col(74) = Ns(1)*Ns(2)*Ns(2)*Ns(3)*dNs.col(0) +
                             Ns(0)*Ns(2)*Ns(3)*Ns(3)*dNs.col(1) +
                             Ns(0)*Ns(1)*Ns(3)*Ns(0)*dNs.col(2) +
                             Ns(0)*Ns(1)*Ns(2)*Ns(1)*dNs.col(3);
                // 4th ord volume functions (interior)
                Nv.col(75) = -Ns(0)*Ns(1)*Ns(2)*Ns(3)*(dNs.col(0)-dNs.col(1)+dNs.col(2)-dNs.col(3));
                Nv.col(76) = Ns(0)*Ns(1)*Ns(2)*Ns(3)*(dNs.col(0)+dNs.col(1)-dNs.col(2)-dNs.col(3));
                Nv.col(77) = Ns(0)*Ns(1)*Ns(2)*Ns(3)*(dNs.col(0)-dNs.col(1)-dNs.col(2)+dNs.col(3));
                Nv.col(78) = Ns(0)*Ns(1)*Ns(2)*Ns(3)*(2*dNs.col(0)-dNs.col(1)-dNs.col(2));
                Nv.col(79) = Ns(0)*Ns(1)*Ns(2)*Ns(3)*(2*dNs.col(1)-dNs.col(0)-dNs.col(2));
                Nv.col(80) = Ns(0)*Ns(1)*Ns(2)*Ns(3)*(2*dNs.col(2)-dNs.col(0)-dNs.col(1));
                Nv.col(81) = Ns(0)*Ns(1)*Ns(2)*Ns(3)*(2*dNs.col(3)-dNs.col(0)-dNs.col(1));
                Nv.col(82) = Ns(0)*Ns(1)*Ns(2)*Ns(3)*(dNs.col(0)+dNs.col(1)+dNs.col(2)-3*dNs.col(3));
                Nv.col(83) = Ns(0)*Ns(1)*Ns(2)*Ns(3)*(dNs.col(0)+dNs.col(2)-2*dNs.col(1));
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
                // 4th ord face curls: face 0-1-2
                dNv.col(51) = Ns(1)*Ns(2)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(0)*Ns(2)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(0)*Ns(1)*cross(dNs.col(2),dNs.col(3));
                dNv.col(52) = Ns(2)*(Ns(0)+Ns(1))*(Ns(1)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(0)*cross(dNs.col(1),dNs.col(3))) +
                              Ns(0)*Ns(1)*(cross(dNs.col(0),dNs.col(3)) +
                              cross(dNs.col(1),dNs.col(3)));
                dNv.col(53) = Ns(2)*(Ns(2)-Ns(1))*(Ns(1)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(0)*cross(dNs.col(1),dNs.col(3))) +
                              Ns(0)*Ns(1)*(Ns(2)-Ns(1))*cross(dNs.col(2),dNs.col(3));
                dNv.col(54) = Ns(0)*Ns(1)*(Ns(0)-Ns(2))*cross(dNs.col(2),dNs.col(3));
                dNv.col(55) = 2*Ns(0)*Ns(1)*Ns(2)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(0)*Ns(0)*Ns(2)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(0)*Ns(0)*Ns(1)*cross(dNs.col(2),dNs.col(3));
                dNv.col(56) = Ns(1)*Ns(1)*Ns(2)*cross(dNs.col(0),dNs.col(3)) +
                              2*Ns(0)*Ns(1)*Ns(2)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(0)*Ns(1)*Ns(1)*cross(dNs.col(2),dNs.col(3));
                // face 0-1-3
                dNv.col(57) = Ns(0)*Ns(1)*cross(dNs.col(2),dNs.col(3));
                dNv.col(58) = Ns(0)*Ns(1)*(Ns(0)+Ns(1))*(cross(dNs.col(2),dNs.col(3)));
                dNv.col(59) = Ns(0)*Ns(1)*(Ns(3)-Ns(1))*(cross(dNs.col(2),dNs.col(3)));
                dNv.col(60) = Ns(0)*Ns(1)*(Ns(0)-Ns(3))*(cross(dNs.col(2),dNs.col(3)));
                dNv.col(61) = 2*Ns(0)*Ns(1)*Ns(0)*cross(dNs.col(2),dNs.col(3)) +
                              Ns(0)*Ns(1)*Ns(1)*0;
                dNv.col(61) = 2*Ns(0)*Ns(1)*Ns(3)*cross(dNs.col(2),dNs.col(3)) +
                              Ns(0)*Ns(0)*Ns(3)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(1)*Ns(3)*Ns(1)*cross(dNs.col(0),dNs.col(3));
                dNv.col(62) = Ns(0)*Ns(1)*Ns(3)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(0)*Ns(1)*Ns(3)*cross(dNs.col(1),dNs.col(3));
                // face 0-2-3
                dNv.col(63) = Ns(0)*Ns(2)*cross(dNs.col(1),dNs.col(3));
                dNv.col(64) = Ns(0)*Ns(2)*(Ns(0)+Ns(3))*(cross(dNs.col(1),dNs.col(3)));
                dNv.col(65) = Ns(0)*Ns(2)*(Ns(3)-Ns(2))*(cross(dNs.col(1),dNs.col(3)));
                dNv.col(66) = Ns(0)*Ns(2)*(Ns(0)-Ns(3))*(cross(dNs.col(1),dNs.col(3)));
                dNv.col(67) = 2*Ns(0)*Ns(2)*Ns(3)*cross(dNs.col(1),dNs.col(3));
                dNv.col(68) = Ns(0)*Ns(2)*Ns(2)*cross(dNs.col(1),dNs.col(3)) +
                              Ns(0)*Ns(2)*Ns(3)*cross(dNs.col(1),dNs.col(3));
                // face 1-2-3
                dNv.col(69) = Ns(1)*Ns(2)*cross(dNs.col(0),dNs.col(3));
                dNv.col(70) = Ns(1)*Ns(2)*(Ns(2)+Ns(3))*(cross(dNs.col(0),dNs.col(3)));
                dNv.col(71) = Ns(1)*Ns(2)*(Ns(3)-Ns(2))*(cross(dNs.col(0),dNs.col(3)));
                dNv.col(72) = Ns(1)*Ns(2)*(Ns(1)-Ns(3))*(cross(dNs.col(0),dNs.col(3)));
                dNv.col(73) = 2*Ns(1)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(3));
                dNv.col(74) = Ns(1)*Ns(2)*Ns(2)*cross(dNs.col(0),dNs.col(3));
                // 4th ord volume curls
                dNv.col(75) = Ns(1)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(1)) +
                              Ns(0)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(1));
                dNv.col(76) = Ns(1)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(1)) +
                              Ns(0)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(1));
                dNv.col(77) = -Ns(0)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(1)) +
                              Ns(0)*Ns(1)*Ns(2)*cross(dNs.col(0),dNs.col(3));
                dNv.col(78) = Ns(1)*Ns(2)*Ns(3)*cross(dNs.col(1),dNs.col(2)) +
                              Ns(0)*Ns(1)*Ns(3)*cross(dNs.col(0),dNs.col(2));
                dNv.col(79) = Ns(0)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(2)) +
                              Ns(0)*Ns(1)*Ns(2)*cross(dNs.col(1),dNs.col(3));
                dNv.col(80) = Ns(1)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(2)) +
                              Ns(0)*Ns(1)*Ns(3)*cross(dNs.col(1),dNs.col(3));
                dNv.col(81) = Ns(0)*Ns(1)*Ns(3)*cross(dNs.col(0),dNs.col(2)) +
                              Ns(1)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(2));
                dNv.col(82) = Ns(0)*Ns(1)*Ns(2)*cross(dNs.col(0),dNs.col(3)) +
                              Ns(0)*Ns(1)*Ns(3)*cross(dNs.col(0),dNs.col(3));
                dNv.col(83) = Ns(0)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(2)) +
                              Ns(1)*Ns(2)*Ns(3)*cross(dNs.col(0),dNs.col(1));
                break;
            default:
                throw std::runtime_error("shapehcurlTetra order not yet implemented");
            }
            break;
        case hgrad:
            switch(p_ord)
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
                // hierarchical basis functions are not appropriate for hgrad space
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
            case 4:
                Ns.resize(1,35);
                dNs.resize(3,35);
                dNs.fill(0);
                cPos.resize(4);
                cPos(3) = 1-cPos(0)-cPos(1)-cPos(2);
                // vertex functions (p=4)
                Ns(0) = (cPos(3)*(4*cPos(3)-1)*(4*cPos(3)-2)*(4*cPos(3)-3))/6;
                Ns(1) = (cPos(0)*(4*cPos(0)-1)*(4*cPos(0)-2)*(4*cPos(0)-3))/6;
                Ns(2) = (cPos(1)*(4*cPos(1)-1)*(4*cPos(1)-2)*(4*cPos(1)-3))/6;
                Ns(3) = (cPos(2)*(4*cPos(2)-1)*(4*cPos(2)-2)*(4*cPos(2)-3))/6;
                // edge functions (p=2 level: 4*lam_i*lam_j)
                Ns(4) = 4*cPos(0)*cPos(1);
                Ns(5) = 4*cPos(0)*cPos(2);
                Ns(6) = 4*cPos(0)*cPos(3);
                Ns(7) = 4*cPos(1)*cPos(2);
                Ns(8) = 4*cPos(1)*cPos(3);
                Ns(9) = 4*cPos(2)*cPos(3);
                // edge functions (p=3 level A: (9/2)*lam_i*lam_j*(3*lam_i-1))
                Ns(10) = (9*cPos(0)*cPos(1)*(3*cPos(0)-1))/2;
                Ns(11) = (9*cPos(0)*cPos(2)*(3*cPos(0)-1))/2;
                Ns(12) = (9*cPos(0)*cPos(3)*(3*cPos(0)-1))/2;
                Ns(13) = (9*cPos(1)*cPos(2)*(3*cPos(1)-1))/2;
                Ns(14) = (9*cPos(1)*cPos(3)*(3*cPos(1)-1))/2;
                Ns(15) = (9*cPos(2)*cPos(3)*(3*cPos(2)-1))/2;
                // edge functions (p=3 level B: (9/2)*lam_i*lam_j*(3*lam_j-1))
                Ns(16) = (9*cPos(0)*cPos(1)*(3*cPos(1)-1))/2;
                Ns(17) = (9*cPos(0)*cPos(2)*(3*cPos(2)-1))/2;
                Ns(18) = (9*cPos(0)*cPos(3)*(3*cPos(3)-1))/2;
                Ns(19) = (9*cPos(1)*cPos(2)*(3*cPos(2)-1))/2;
                Ns(20) = (9*cPos(1)*cPos(3)*(3*cPos(3)-1))/2;
                Ns(21) = (9*cPos(2)*cPos(3)*(3*cPos(3)-1))/2;
                // face functions (p=3 level: 27*lam_i*lam_j*lam_k)
                Ns(22) = 27*cPos(1)*cPos(2)*cPos(3);  // face opposite vtx 0
                Ns(23) = 27*cPos(0)*cPos(2)*cPos(3);  // face opposite vtx 1
                Ns(24) = 27*cPos(0)*cPos(1)*cPos(3);  // face opposite vtx 2
                Ns(25) = 27*cPos(0)*cPos(1)*cPos(2);  // face opposite vtx 3
                // face functions (p=4 level A: lam_i*lam_j*lam_k*(lam_j-lam_i))
                Ns(26) = cPos(1)*cPos(2)*cPos(3)*(cPos(2)-cPos(1));
                Ns(27) = cPos(0)*cPos(2)*cPos(3)*(cPos(2)-cPos(0));
                Ns(28) = cPos(0)*cPos(1)*cPos(3)*(cPos(1)-cPos(0));
                Ns(29) = cPos(0)*cPos(1)*cPos(2)*(cPos(1)-cPos(0));
                // face functions (p=4 level B: lam_i*lam_j*lam_k*(lam_i+lam_j-2*lam_k))
                Ns(30) = cPos(1)*cPos(2)*cPos(3)*(cPos(1)+cPos(2)-2*cPos(3));
                Ns(31) = cPos(0)*cPos(2)*cPos(3)*(cPos(0)+cPos(2)-2*cPos(3));
                Ns(32) = cPos(0)*cPos(1)*cPos(3)*(cPos(0)+cPos(1)-2*cPos(3));
                Ns(33) = cPos(0)*cPos(1)*cPos(2)*(cPos(0)+cPos(1)-2*cPos(2));
                // volume function (p=4: 256*lam_0*lam_1*lam_2*lam_3)
                Ns(34) = 256*cPos(0)*cPos(1)*cPos(2)*cPos(3);
                // derivative of vertex functions
                { double d = (128*pow(cPos(3),3) - 144*pow(cPos(3),2) + 44*cPos(3) - 3)/3;
                dNs(0,0) = -d; dNs(1,0) = -d; dNs(2,0) = -d; }
                dNs(0,1) = (128*pow(cPos(0),3) - 144*pow(cPos(0),2) + 44*cPos(0) - 3)/3;
                dNs(1,2) = (128*pow(cPos(1),3) - 144*pow(cPos(1),2) + 44*cPos(1) - 3)/3;
                dNs(2,3) = (128*pow(cPos(2),3) - 144*pow(cPos(2),2) + 44*cPos(2) - 3)/3;
                // derivatives of edge functions (p=2 level: 4*lam_i*lam_j)
                dNs(0,4) = 4*cPos(1); dNs(1,4) = 4*cPos(0);
                dNs(0,5) = 4*cPos(2); dNs(2,5) = 4*cPos(0);
                dNs(0,6) = 4*(cPos(3)-cPos(0)); dNs(1,6) = -4*cPos(0); dNs(2,6) = -4*cPos(0);
                dNs(1,7) = 4*cPos(2); dNs(2,7) = 4*cPos(1);
                dNs(0,8) = -4*cPos(1); dNs(1,8) = 4*(cPos(3)-cPos(1)); dNs(2,8) = -4*cPos(1);
                dNs(0,9) = -4*cPos(2); dNs(1,9) = -4*cPos(2); dNs(2,9) = 4*(cPos(3)-cPos(2));
                // derivatives of edge functions (p=3 level A: (9/2)*lam_i*lam_j*(3*lam_i-1))
                dNs(0,10) = (9*cPos(1)*(6*cPos(0)-1))/2;
                dNs(1,10) = (9*cPos(0)*(3*cPos(0)-1))/2;
                dNs(0,11) = (9*cPos(2)*(6*cPos(0)-1))/2;
                dNs(2,11) = (9*cPos(0)*(3*cPos(0)-1))/2;
                dNs(0,12) = (9*(cPos(3)*(6*cPos(0)-1)-cPos(0)*(3*cPos(0)-1)))/2;
                dNs(1,12) = -(9*cPos(0)*(3*cPos(0)-1))/2;
                dNs(2,12) = -(9*cPos(0)*(3*cPos(0)-1))/2;
                dNs(1,13) = (9*cPos(2)*(6*cPos(1)-1))/2;
                dNs(2,13) = (9*cPos(1)*(3*cPos(1)-1))/2;
                dNs(0,14) = -(9*cPos(1)*(3*cPos(1)-1))/2;
                dNs(1,14) = (9*(cPos(3)*(6*cPos(1)-1)-cPos(1)*(3*cPos(1)-1)))/2;
                dNs(2,14) = -(9*cPos(1)*(3*cPos(1)-1))/2;
                dNs(0,15) = -(9*cPos(2)*(3*cPos(2)-1))/2;
                dNs(1,15) = -(9*cPos(2)*(3*cPos(2)-1))/2;
                dNs(2,15) = (9*(cPos(3)*(6*cPos(2)-1)-cPos(2)*(3*cPos(2)-1)))/2;
                // derivatives of edge functions (p=3 level B: (9/2)*lam_i*lam_j*(3*lam_j-1))
                dNs(0,16) = (9*cPos(1)*(3*cPos(1)-1))/2;
                dNs(1,16) = (9*cPos(0)*(6*cPos(1)-1))/2;
                dNs(0,17) = (9*cPos(2)*(3*cPos(2)-1))/2;
                dNs(2,17) = (9*cPos(0)*(6*cPos(2)-1))/2;
                dNs(0,18) = (9*(cPos(3)*(3*cPos(3)-1)-cPos(0)*(6*cPos(3)-1)))/2;
                dNs(1,18) = -(9*cPos(0)*(6*cPos(3)-1))/2;
                dNs(2,18) = -(9*cPos(0)*(6*cPos(3)-1))/2;
                dNs(1,19) = (9*cPos(2)*(3*cPos(2)-1))/2;
                dNs(2,19) = (9*cPos(1)*(6*cPos(2)-1))/2;
                dNs(0,20) = -(9*cPos(1)*(6*cPos(3)-1))/2;
                dNs(1,20) = (9*(cPos(3)*(3*cPos(3)-1)-cPos(1)*(6*cPos(3)-1)))/2;
                dNs(2,20) = -(9*cPos(1)*(6*cPos(3)-1))/2;
                dNs(0,21) = -(9*cPos(2)*(6*cPos(3)-1))/2;
                dNs(1,21) = -(9*cPos(2)*(6*cPos(3)-1))/2;
                dNs(2,21) = (9*(cPos(3)*(3*cPos(3)-1)-cPos(2)*(6*cPos(3)-1)))/2;
                // derivatives of face functions (p=3 level: 27*lam_i*lam_j*lam_k)
                dNs(0,22) = -27*cPos(1)*cPos(2);
                dNs(1,22) = 27*cPos(2)*(cPos(3)-cPos(1));
                dNs(2,22) = 27*cPos(1)*(cPos(3)-cPos(2));
                dNs(0,23) = 27*cPos(2)*(cPos(3)-cPos(0));
                dNs(1,23) = -27*cPos(0)*cPos(2);
                dNs(2,23) = 27*cPos(0)*(cPos(3)-cPos(2));
                dNs(0,24) = 27*cPos(1)*(cPos(3)-cPos(0));
                dNs(1,24) = 27*cPos(0)*(cPos(3)-cPos(1));
                dNs(2,24) = -27*cPos(0)*cPos(1);
                dNs(0,25) = 27*cPos(1)*cPos(2);
                dNs(1,25) = 27*cPos(0)*cPos(2);
                dNs(2,25) = 27*cPos(0)*cPos(1);
                // derivatives of face functions (p=4 level A: lam_i*lam_j*lam_k*(lam_j-lam_i))
                dNs(0,26) = -cPos(1)*cPos(2)*cPos(3)*(cPos(2)-cPos(1));
                dNs(1,26) = cPos(2)*(cPos(3)*(cPos(2)-2*cPos(1))-cPos(1)*(cPos(2)-cPos(1)));
                dNs(2,26) = cPos(1)*(cPos(3)*(2*cPos(2)-cPos(1))-cPos(2)*(cPos(2)-cPos(1)));
                dNs(0,27) = cPos(2)*cPos(3)*(cPos(2)-2*cPos(0))-cPos(0)*cPos(2)*(cPos(2)-cPos(0));
                dNs(1,27) = -cPos(0)*cPos(2)*(cPos(2)-cPos(0));
                dNs(2,27) = cPos(0)*cPos(3)*(2*cPos(2)-cPos(0))-cPos(0)*cPos(2)*(cPos(2)-cPos(0));
                dNs(0,28) = cPos(1)*cPos(3)*(cPos(1)-2*cPos(0))-cPos(0)*cPos(1)*(cPos(1)-cPos(0));
                dNs(1,28) = cPos(0)*cPos(3)*(2*cPos(1)-cPos(0))-cPos(0)*cPos(1)*(cPos(1)-cPos(0));
                dNs(2,28) = -cPos(0)*cPos(1)*(cPos(1)-cPos(0));
                dNs(0,29) = cPos(1)*cPos(2)*(cPos(1)-2*cPos(0));
                dNs(1,29) = cPos(0)*cPos(2)*(2*cPos(1)-cPos(0));
                dNs(2,29) = cPos(0)*cPos(1)*(cPos(1)-cPos(0));
                // derivatives of face functions (p=4 level B: lam_i*lam_j*lam_k*(lam_i+lam_j-2*lam_k))
                dNs(0,30) = -cPos(1)*cPos(2)*(cPos(1)+cPos(2)-4*cPos(3));
                dNs(1,30) = cPos(2)*cPos(3)*(2*cPos(1)+cPos(2)-2*cPos(3))-cPos(1)*cPos(2)*(cPos(1)+cPos(2)-4*cPos(3));
                dNs(2,30) = cPos(1)*cPos(3)*(cPos(1)+2*cPos(2)-2*cPos(3))-cPos(1)*cPos(2)*(cPos(1)+cPos(2)-4*cPos(3));
                dNs(0,31) = cPos(2)*cPos(3)*(2*cPos(0)+cPos(2)-2*cPos(3))-cPos(0)*cPos(2)*(cPos(0)+cPos(2)-4*cPos(3));
                dNs(1,31) = -cPos(0)*cPos(2)*(cPos(0)+cPos(2)-4*cPos(3));
                dNs(2,31) = cPos(0)*cPos(3)*(cPos(0)+2*cPos(2)-2*cPos(3))-cPos(0)*cPos(2)*(cPos(0)+cPos(2)-4*cPos(3));
                dNs(0,32) = cPos(1)*cPos(3)*(2*cPos(0)+cPos(1)-2*cPos(3))-cPos(0)*cPos(1)*(cPos(0)+cPos(1)-4*cPos(3));
                dNs(1,32) = cPos(0)*cPos(3)*(cPos(0)+2*cPos(1)-2*cPos(3))-cPos(0)*cPos(1)*(cPos(0)+cPos(1)-4*cPos(3));
                dNs(2,32) = -cPos(0)*cPos(1)*(cPos(0)+cPos(1)-4*cPos(3));
                dNs(0,33) = cPos(1)*cPos(2)*(2*cPos(0)+cPos(1)-2*cPos(2));
                dNs(1,33) = cPos(0)*cPos(2)*(cPos(0)+2*cPos(1)-2*cPos(2));
                dNs(2,33) = cPos(0)*cPos(1)*(cPos(0)+cPos(1)-4*cPos(2));
                // derivative of volume function (256*lam_0*lam_1*lam_2*lam_3)
                dNs(0,34) = 256*cPos(1)*cPos(2)*(cPos(3)-cPos(0));
                dNs(1,34) = 256*cPos(0)*cPos(2)*(cPos(3)-cPos(1));
                dNs(2,34) = 256*cPos(0)*cPos(1)*(cPos(3)-cPos(2));
                dNs = cJac->invJ * dNs;
                break;
            default:
                throw std::runtime_error("shapehgradTetra order not yet implemented");
            }
            break;
        case hdiv:
            // H(div) Raviart-Thomas (RT) basis on tetrahedron
            // Uses reference gradients, then applies H(div) Piola transform:
            //   Nv = (1/detJ) * J * hat_Nv_ref
            //   div(Nv) = (1/detJ) * div_ref(hat_Nv_ref)
            {
                // Reference gradients of barycentric coordinates (before invJ mapping)
                arma::mat dNs_ref(3, 4);
                dNs_ref(0,0) = -1; dNs_ref(1,0) = -1; dNs_ref(2,0) = -1;
                dNs_ref(0,1) =  1; dNs_ref(0,2) =  0; dNs_ref(0,3) =  0;
                dNs_ref(1,1) =  0; dNs_ref(1,2) =  1; dNs_ref(1,3) =  0;
                dNs_ref(2,1) =  0; dNs_ref(2,2) =  0; dNs_ref(2,3) =  1;
                // Ns: barycentric coordinates (size 1x4)
                // dNs will be overwritten by invJ mapping for physical gradients
                Ns.resize(1,4);
                dNs.resize(3,4);
                Ns(0) = 1 - cPos(0) - cPos(1) - cPos(2);
                Ns(1) = cPos(0);
                Ns(2) = cPos(1);
                Ns(3) = cPos(2);
                dNs(0,0) = -1; dNs(1,0) = -1; dNs(2,0) = -1;
                dNs(0,1) =  1;
                dNs(1,2) =  1;
                dNs(2,3) =  1;
                dNs = cJac->invJ * dNs;  // physical gradients
                switch(p_ord)
                {
                case 1:
                {
                    // RT0: 4 face basis functions on tetrahedron
                    // hat_RT_k = (x - v_k) / (3*|K|) on ref tet (|K|=1/6)
                    //         = 2 * (x - v_k)
                    // Piola: Nv_k = (1/detJ) * J * hat_RT_k,  div(Nv_k) = 6/detJ
                    Nv.resize(3, 4);
                    divNv.resize(1, 4);
                    arma::mat J = arma::inv(cJac->invJ);
                    arma::mat hat_Nv(3, 4);
                    for (int k = 0; k < 4; k++) {
                        hat_Nv(0, k) = 2.0 * (cPos(0) - (k == 1 ? 1.0 : 0.0));
                        hat_Nv(1, k) = 2.0 * (cPos(1) - (k == 2 ? 1.0 : 0.0));
                        hat_Nv(2, k) = 2.0 * (cPos(2) - (k == 3 ? 1.0 : 0.0));
                    }
                    Nv = (1.0 / cJac->detJ) * J * hat_Nv;
                    divNv.fill(6.0 / cJac->detJ);
                    break;
                }
                default:
                    throw std::runtime_error("shapehdivTetra order not yet implemented");
                }
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
        if(sType == hdiv) {
            // H(div) Raviart-Thomas (RT) basis on triangle (2D)
            // Reference triangle: v0=(0,0), v1=(1,0), v2=(0,1)
            // RT0 on triangle: 3 edge functions, each with unit normal flux through one edge.
            // hat_RT_k = (x - v_k) / (2*|K|) where |K| = 1/2 for ref triangle.
            // So: hat_RT_k = (x - v_k) in reference coordinates.
            // hat_RT_0 = (x, y) = (lam_1, lam_2)
            // hat_RT_1 = (x-1, y) = (lam_1-1, lam_2)
            // hat_RT_2 = (x, y-1) = (lam_1, lam_2-1)
            // div_ref(hat_RT_k) = 2 for all k.
            //
            // Reference triangle: K={(x,y): x>=0, y>=0, x+y<=1}
            // Edge 0 (v1-v2, opposite v0): x+y=1, outward normal n0=(1,1)/√2, |e0|=√2
            // Edge 1 (v0-v2, opposite v1): x=0, outward normal n1=(-1,0), |e1|=1
            // Edge 2 (v0-v1, opposite v2): y=0, outward normal n2=(0,-1), |e2|=1
            //
            // hat_RT_0 = (x,y): flux through edge 0 = ∫(x+y)/√2 dS = (1)/√2 * √2 = 1 ✓
            // on edge 1 (x=0): hat_RT_0 · (-1,0) = -x = 0 ✓
            // on edge 2 (y=0): hat_RT_0 · (0,-1) = -y = 0 ✓
            // hat_RT_1 = (x-1,y): on edge 0: (x-1+y)/√2 = 0 ✓
            // on edge 1 (x=0): (x-1,y)·(-1,0) = -(x-1)=1 at x=0, ∫ 1 dS = 1 ✓
            // on edge 2 (y=0): (x-1,y)·(0,-1) = -y = 0 ✓
            // hat_RT_2 = (x,y-1): on edge 0: (x+y-1)/√2 = 0 ✓
            // on edge 1: (x,y-1)·(-1,0) = -x = 0 at x=0 ✓
            // on edge 2: (x,y-1)·(0,-1) = -(y-1) = 1 at y=0, ∫ 1 dS = 1 ✓
            // div_ref = 2 for all ✓

            arma::mat J = arma::inv(cJac->invJ);
            arma::mat hat_Nv_ref(2, 3);
            switch(p_ord) {
            case 1:
                Nv.resize(2, 3);
                divNv.resize(1, 3);
                // Reference RT0 basis in terms of cPos=(lam_1, lam_2):
                // hat_RT_2 = (lam_1, lam_2)         edge (1,2) with outward normal (1,1)/sqrt(2)
                // hat_RT_1 = (lam_1-1, lam_2)       edge (0,2) with outward normal (-1,0)
                // hat_RT_0 = (lam_1, lam_2-1)       edge (0,1) with outward normal (0,-1)
                // Reordered to i<j: col 0 = edge (0,1), col 1 = edge (0,2), col 2 = edge (1,2)
                for(int k = 0; k < 3; k++) {
                    hat_Nv_ref(0, k) = cPos(0) - (k == 2 ? 1.0 : 0.0);
                    hat_Nv_ref(1, k) = cPos(1) - (k == 0 ? 1.0 : 0.0);
                }
                // 2D Piola transform: Nv = (1/detJ) * J * hat_Nv_ref
                Nv = (1.0 / cJac->detJ) * J * hat_Nv_ref;
                // div(Nv) = (1/detJ) * div_ref(hat_Nv) = 2/detJ
                divNv.fill(2.0 / cJac->detJ);
                break;
            default:
                throw std::runtime_error("shapehdivTria order not yet implemented");
            }
            break;  // break out of case 2: switch — skip hcurl computation below
        }
        // hcurl computation (existing): Nv, dNv for Nedelec elements on triangle
        switch(p_ord)
        {
        case 1:
            Nv.resize(2,3);
            dNv.resize(1,3);
            cNv.resize(2,3);
            Nv.col(0) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
            Nv.col(1) = Ns(2)*dNs.col(0)-Ns(0)*dNs.col(2);
            Nv.col(2) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
            cNv(0,0) = Ns(1);
            cNv(1,0) = -Ns(2)+1.0;
            cNv(0,1) = -Ns(1)+1.0;
            cNv(1,1) = Ns(2);
            cNv(0,2) = -std::sqrt(2)*Ns(1);
            cNv(1,2) = -std::sqrt(2)*Ns(2);
            dNv(0,0) = 2;
            dNv(0,1) = -2;
            dNv(0,2) = 2;
            cNv *= cNs(0,1)*cNs(1,2)-cNs(1,1)*cNs(0,2);
            dNv *= dNs(0,1)*dNs(1,2)-dNs(1,1)*dNs(0,2);
            break;
        case 2:
            Nv.resize(2,8);
            dNv.resize(1,8);
            Nv.col(0) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
            Nv.col(1) = Ns(2)*dNs.col(0)-Ns(0)*dNs.col(2);
            Nv.col(2) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
            Nv.col(3) = 4*(Ns(0)*dNs.col(1)+Ns(1)*dNs.col(0));
            Nv.col(4) = 4*(Ns(0)*dNs.col(2)+Ns(2)*dNs.col(0));
            Nv.col(5) = 4*(Ns(1)*dNs.col(2)+Ns(2)*dNs.col(1));
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
            Ns(4) = 4*cPos(0)*cPos(2);
            Ns(5) = 4*cPos(2)*cPos(1);
            dNs(0,0) = -1;
            dNs(1,0) = -1;
            dNs(0,1) = 1;
            dNs(1,2) = 1;
            dNs(0,3) = 4*cPos(1);
            dNs(0,4) = 4*(cPos(2)-cPos(0));
            dNs(0,5) = -4*cPos(1);
            dNs(1,3) = 4*cPos(0);
            dNs(1,4) = -4*cPos(0);
            dNs(1,5) = 4*(cPos(2)-cPos(1));
            dNs = cJac->invJ * dNs;
            break;
        case 3:
            Nv.resize(2,15);
            dNv.resize(1,15);
            Nv.col(0) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
            Nv.col(1) = Ns(2)*dNs.col(0)-Ns(0)*dNs.col(2);
            Nv.col(2) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
            Nv.col(3) = 4*(Ns(0)*dNs.col(1)+Ns(1)*dNs.col(0));
            Nv.col(4) = 4*(Ns(0)*dNs.col(2)+Ns(2)*dNs.col(0));
            Nv.col(5) = 4*(Ns(1)*dNs.col(2)+Ns(2)*dNs.col(1));
            Nv.col(6) = Ns(0)*Ns(1)*dNs.col(2)-Ns(0)*Ns(2)*dNs.col(1);
            Nv.col(7) = Ns(0)*Ns(1)*dNs.col(2)-Ns(1)*Ns(2)*dNs.col(0);
            Nv.col(8) = Ns(0)*(Ns(0)-2*Ns(1))*dNs.col(1) +
                         Ns(1)*(2*Ns(0)-Ns(1))*dNs.col(0);
            Nv.col(9) = Ns(0)*(Ns(0)-2*Ns(2))*dNs.col(2) +
                        Ns(2)*(2*Ns(0)-Ns(2))*dNs.col(0);
            Nv.col(10) = Ns(1)*(Ns(1)-2*Ns(2))*dNs.col(2) +
                         Ns(2)*(2*Ns(1)-Ns(2))*dNs.col(1);
            Nv.col(11) = Ns(1)*Ns(0)*dNs.col(2)+
                         Ns(0)*Ns(2)*dNs.col(1)+
                         Ns(1)*Ns(2)*dNs.col(0);
            Nv.col(12) = -Ns(0)*Ns(1)*(Ns(0)-2*Ns(2))*dNs.col(2) -
                         Ns(1)*Ns(2)*(2*Ns(0)-Ns(2))*dNs.col(0) +
                         3*Ns(0)*Ns(2)*(Ns(0)-Ns(2))*dNs.col(1);
            Nv.col(13) = -Ns(1)*Ns(0)*(Ns(1)-2*Ns(2))*dNs.col(2) -
                         Ns(0)*Ns(2)*(2*Ns(1)-Ns(2))*dNs.col(1) +
                         3*Ns(1)*Ns(2)*(Ns(1)-Ns(2))*dNs.col(0);
            Nv.col(14) = -Ns(0)*Ns(2)*(Ns(0)-2*Ns(1))*dNs.col(1) -
                         Ns(1)*Ns(2)*(2*Ns(0)-Ns(1))*dNs.col(0) +
                         3*Ns(0)*Ns(1)*(Ns(0)-Ns(1))*dNs.col(2);
            dNv(0,0) = 2;
            dNv(0,1) = -2;
            dNv(0,2) = 2;
            dNv(0,6) = 2*Ns(0)-Ns(1)-Ns(2);
            dNv(0,7) = Ns(0)+Ns(2)-2*Ns(1);
            dNv(0,12) = -4*Ns(0)*(Ns(0)-2*Ns(2)) +
                        4*Ns(2)*(2*Ns(0)-Ns(2));
            dNv(0,13) = 4*Ns(1)*(Ns(1)-2*Ns(2)) -
                        4*Ns(2)*(2*Ns(1)-Ns(2));
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
            Ns(4) = 4*cPos(0)*cPos(2);
            Ns(5) = 4*cPos(2)*cPos(1);
            Ns(6) = cPos(0)*cPos(1)*(cPos(0)-cPos(1));
            Ns(7) = cPos(2)*cPos(0)*(cPos(2)-cPos(0));
            Ns(8) = cPos(2)*cPos(1)*(cPos(2)-cPos(1));
            Ns(9) = cPos(2)*cPos(0)*cPos(1);
            dNs(0,0) = -1;
            dNs(1,0) = -1;
            dNs(0,1) = 1;
            dNs(1,2) = 1;
            dNs(0,3) = 4*cPos(1);
            dNs(0,4) = 4*(cPos(2)-cPos(0));
            dNs(0,5) = -4*cPos(1);
            dNs(1,3) = 4*cPos(0);
            dNs(1,4) = -4*cPos(0);
            dNs(1,5) = 4*(cPos(2)-cPos(1));
            dNs(0,6) = cPos(1)*(cPos(0)-cPos(1)) + cPos(0)*cPos(1);
            dNs(0,7) = cPos(2)*(cPos(2)-cPos(0)) - cPos(2)*cPos(0) - cPos(0)*(cPos(2)-cPos(0)) - cPos(2)*cPos(0);
            dNs(0,8) = -cPos(1)*(cPos(2)-cPos(1)) - cPos(2)*cPos(1);
            dNs(0,9) = (cPos(2) - cPos(0))*cPos(1);
            dNs(1,6) = cPos(0)*(cPos(0)-cPos(1)) - cPos(0)*cPos(1);
            dNs(1,7) = -cPos(0)*(cPos(2)-cPos(0)) - cPos(2)*cPos(0);
            dNs(1,8) = cPos(2)*(cPos(2)-cPos(1)) - cPos(2)*cPos(1) - cPos(1)*(cPos(2)-cPos(1)) - cPos(2)*cPos(1);
            dNs(1,9) = (cPos(2) - cPos(1))*cPos(0);
            dNs = cJac->invJ * dNs;
            break;
        case 4:
            Nv.resize(2,24);
            dNv.resize(1,24);
            // edge level 1 (p=1)
            Nv.col(0) = Ns(0)*dNs.col(1)-Ns(1)*dNs.col(0);
            Nv.col(1) = Ns(2)*dNs.col(0)-Ns(0)*dNs.col(2);
            Nv.col(2) = Ns(1)*dNs.col(2)-Ns(2)*dNs.col(1);
            // edge level 2 (p=2: 4*(lam_i*dlam_j + lam_j*dlam_i))
            Nv.col(3) = 4*(Ns(0)*dNs.col(1)+Ns(1)*dNs.col(0));
            Nv.col(4) = 4*(Ns(0)*dNs.col(2)+Ns(2)*dNs.col(0));
            Nv.col(5) = 4*(Ns(1)*dNs.col(2)+Ns(2)*dNs.col(1));
            // face level A (p=2)
            Nv.col(6) = Ns(0)*Ns(1)*dNs.col(2)-Ns(0)*Ns(2)*dNs.col(1);
            Nv.col(7) = Ns(0)*Ns(1)*dNs.col(2)-Ns(1)*Ns(2)*dNs.col(0);
            // edge level 3 (p=3: lam_i*(lam_i-2*lam_j)*dlam_j + lam_j*(2*lam_i-lam_j)*dlam_i)
            Nv.col(8) = Ns(0)*(Ns(0)-2*Ns(1))*dNs.col(1) +
                         Ns(1)*(2*Ns(0)-Ns(1))*dNs.col(0);
            Nv.col(9) = Ns(0)*(Ns(0)-2*Ns(2))*dNs.col(2) +
                        Ns(2)*(2*Ns(0)-Ns(2))*dNs.col(0);
            Nv.col(10) = Ns(1)*(Ns(1)-2*Ns(2))*dNs.col(2) +
                         Ns(2)*(2*Ns(1)-Ns(2))*dNs.col(1);
            // face level B (p=3)
            Nv.col(11) = Ns(1)*Ns(0)*dNs.col(2)+
                         Ns(0)*Ns(2)*dNs.col(1)+
                         Ns(1)*Ns(2)*dNs.col(0);
            Nv.col(12) = -Ns(0)*Ns(1)*(Ns(0)-2*Ns(2))*dNs.col(2) -
                         Ns(1)*Ns(2)*(2*Ns(0)-Ns(2))*dNs.col(0) +
                         3*Ns(0)*Ns(2)*(Ns(0)-Ns(2))*dNs.col(1);
            Nv.col(13) = -Ns(1)*Ns(0)*(Ns(1)-2*Ns(2))*dNs.col(2) -
                         Ns(0)*Ns(2)*(2*Ns(1)-Ns(2))*dNs.col(1) +
                         3*Ns(1)*Ns(2)*(Ns(1)-Ns(2))*dNs.col(0);
            Nv.col(14) = -Ns(0)*Ns(2)*(Ns(0)-2*Ns(1))*dNs.col(1) -
                         Ns(1)*Ns(2)*(2*Ns(0)-Ns(1))*dNs.col(0) +
                         3*Ns(0)*Ns(1)*(Ns(0)-Ns(1))*dNs.col(2);
            // edge level 4 (p=4: lam_i^2*(lam_i-3*lam_j)*dlam_j + lam_j^2*(lam_j-3*lam_i)*dlam_i)
            Nv.col(15) = Ns(0)*Ns(0)*(Ns(0)-3*Ns(1))*dNs.col(1) +
                         Ns(1)*Ns(1)*(Ns(1)-3*Ns(0))*dNs.col(0);
            Nv.col(16) = Ns(0)*Ns(0)*(Ns(0)-3*Ns(2))*dNs.col(2) +
                         Ns(2)*Ns(2)*(Ns(2)-3*Ns(0))*dNs.col(0);
            Nv.col(17) = Ns(1)*Ns(1)*(Ns(1)-3*Ns(2))*dNs.col(2) +
                         Ns(2)*Ns(2)*(Ns(2)-3*Ns(1))*dNs.col(1);
            // face level C (p=4)
            Nv.col(18) = Ns(0)*Ns(0)*Ns(1)*dNs.col(2)-Ns(0)*Ns(0)*Ns(2)*dNs.col(1);
            Nv.col(19) = Ns(0)*Ns(1)*Ns(1)*dNs.col(2)-Ns(0)*Ns(1)*Ns(2)*dNs.col(1);
            Nv.col(20) = Ns(0)*Ns(1)*Ns(2)*dNs.col(2)-Ns(0)*Ns(2)*Ns(2)*dNs.col(1);
            Nv.col(21) = Ns(0)*Ns(0)*Ns(1)*dNs.col(2)-Ns(0)*Ns(1)*Ns(2)*dNs.col(0);
            Nv.col(22) = Ns(0)*Ns(1)*Ns(1)*dNs.col(2)-Ns(1)*Ns(1)*Ns(2)*dNs.col(0);
            Nv.col(23) = Ns(0)*Ns(1)*Ns(2)*dNs.col(2)-Ns(1)*Ns(2)*Ns(2)*dNs.col(0);
            // dNv: curls
            dNv(0,0) = 2; dNv(0,1) = -2; dNv(0,2) = 2;
            dNv(0,6) = 2*Ns(0)-Ns(1)-Ns(2);
            dNv(0,7) = Ns(0)+Ns(2)-2*Ns(1);
            dNv(0,12) = -4*Ns(0)*(Ns(0)-2*Ns(2)) + 4*Ns(2)*(2*Ns(0)-Ns(2));
            dNv(0,13) = 4*Ns(1)*(Ns(1)-2*Ns(2)) - 4*Ns(2)*(2*Ns(1)-Ns(2));
            dNv(0,14) = 4*Ns(0)*(Ns(0)-2*Ns(1)) - 4*Ns(1)*(2*Ns(0)-Ns(1));
            dNv(0,15) = 3*(Ns(0)*Ns(0)-Ns(1)*Ns(1));
            dNv(0,16) = 3*(Ns(2)*Ns(2)-Ns(0)*Ns(0));
            dNv(0,17) = 3*(Ns(1)*Ns(1)-Ns(2)*Ns(2));
            dNv(0,18) = 2*Ns(0)*Ns(0) - 2*Ns(0)*(Ns(1)+Ns(2));
            dNv(0,19) = -Ns(1)*Ns(1) + 3*Ns(0)*Ns(1) - Ns(1)*Ns(2);
            dNv(0,20) = -Ns(1)*Ns(2) + 3*Ns(0)*Ns(2) - Ns(2)*Ns(2);
            dNv(0,21) = Ns(0)*Ns(0) - 3*Ns(0)*Ns(1) + Ns(0)*Ns(2);
            dNv(0,22) = 2*Ns(0)*Ns(1) + 2*Ns(1)*Ns(2) - 2*Ns(1)*Ns(1);
            dNv(0,23) = Ns(0)*Ns(2) + Ns(2)*Ns(2) - 3*Ns(1)*Ns(2);
            dNv *= dNs(0,1)*dNs(1,2)-dNs(1,1)*dNs(0,2);
            // update scalar functions
            Ns.resize(1,15);
            dNs.resize(2,15);
            dNs.fill(0);
            cPos.resize(3);
            cPos(2) = 1-cPos(0)-cPos(1);
            // vertex functions (p=4)
            Ns(0) = (cPos(2)*(4*cPos(2)-1)*(4*cPos(2)-2)*(4*cPos(2)-3))/6;
            Ns(1) = (cPos(0)*(4*cPos(0)-1)*(4*cPos(0)-2)*(4*cPos(0)-3))/6;
            Ns(2) = (cPos(1)*(4*cPos(1)-1)*(4*cPos(1)-2)*(4*cPos(1)-3))/6;
            // edge functions level A (p=2: 4*lam_i*lam_j)
            Ns(3) = 4*cPos(0)*cPos(1);
            Ns(4) = 4*cPos(0)*cPos(2);
            Ns(5) = 4*cPos(2)*cPos(1);
            // edge functions level B (p=3: lam_i*lam_j*(lam_i-lam_j))
            Ns(6) = cPos(0)*cPos(1)*(cPos(0)-cPos(1));
            Ns(7) = cPos(2)*cPos(0)*(cPos(2)-cPos(0));
            Ns(8) = cPos(2)*cPos(1)*(cPos(2)-cPos(1));
            // edge functions level C (p=4: lam_i*lam_j*(lam_i*lam_i-lam_j*lam_j))
            Ns(9) = cPos(0)*cPos(1)*(cPos(0)*cPos(0)-cPos(1)*cPos(1));
            Ns(10) = cPos(0)*cPos(2)*(cPos(0)*cPos(0)-cPos(2)*cPos(2));
            Ns(11) = cPos(2)*cPos(1)*(cPos(2)*cPos(2)-cPos(1)*cPos(1));
            // face functions
            Ns(12) = cPos(2)*cPos(0)*cPos(1);                // p=3 face
            Ns(13) = cPos(2)*cPos(0)*cPos(1)*(cPos(1)-cPos(0));  // p=4 face A
            Ns(14) = cPos(2)*cPos(0)*cPos(1)*(cPos(0)+cPos(1)-2*cPos(2));  // p=4 face B
            // dNs: vertex derivatives
            { double d = (128*pow(cPos(2),3)-144*pow(cPos(2),2)+44*cPos(2)-3)/3;
            dNs(0,0) = -d; dNs(1,0) = -d; }
            dNs(0,1) = (128*pow(cPos(0),3)-144*pow(cPos(0),2)+44*cPos(0)-3)/3;
            dNs(1,2) = (128*pow(cPos(1),3)-144*pow(cPos(1),2)+44*cPos(1)-3)/3;
            // dNs: edge level A (4*lam_i*lam_j)
            dNs(0,3) = 4*cPos(1); dNs(1,3) = 4*cPos(0);
            dNs(0,4) = 4*(cPos(2)-cPos(0)); dNs(1,4) = -4*cPos(0);
            dNs(0,5) = -4*cPos(1); dNs(1,5) = 4*(cPos(2)-cPos(1));
            // dNs: edge level B (lam_i*lam_j*(lam_i-lam_j))
            dNs(0,6) = cPos(1)*(2*cPos(0)-cPos(1));
            dNs(1,6) = cPos(0)*(cPos(0)-2*cPos(1));
            dNs(0,7) = cPos(2)*(cPos(2)-2*cPos(0)) - cPos(0)*(2*cPos(2)-cPos(0));
            dNs(1,7) = -cPos(0)*(2*cPos(2)-cPos(0));
            dNs(0,8) = -cPos(1)*(2*cPos(2)-cPos(1));
            dNs(1,8) = cPos(2)*(cPos(2)-2*cPos(1)) - cPos(1)*(2*cPos(2)-cPos(1));
            // dNs: edge level C (lam_i*lam_j*(lam_i^2-lam_j^2))
            dNs(0,9) = cPos(1)*(3*cPos(0)*cPos(0)-cPos(1)*cPos(1));
            dNs(1,9) = cPos(0)*(cPos(0)*cPos(0)-3*cPos(1)*cPos(1));
            dNs(0,10) = cPos(2)*(3*cPos(0)*cPos(0)-cPos(2)*cPos(2))-cPos(0)*(cPos(0)*cPos(0)-3*cPos(2)*cPos(2));
            dNs(1,10) = -cPos(0)*(cPos(0)*cPos(0)-3*cPos(2)*cPos(2));
            dNs(0,11) = -cPos(1)*(3*cPos(2)*cPos(2)-cPos(1)*cPos(1));
            dNs(1,11) = cPos(2)*(cPos(2)*cPos(2)-3*cPos(1)*cPos(1))-cPos(1)*(3*cPos(2)*cPos(2)-cPos(1)*cPos(1));
            // dNs: face functions
            dNs(0,12) = cPos(1)*(cPos(2)-cPos(0));
            dNs(1,12) = cPos(0)*(cPos(2)-cPos(1));
            dNs(0,13) = cPos(1)*cPos(2)*(cPos(1)-2*cPos(0))-cPos(0)*cPos(1)*(cPos(1)-cPos(0));
            dNs(1,13) = cPos(0)*cPos(2)*(2*cPos(1)-cPos(0))-cPos(0)*cPos(1)*(cPos(1)-cPos(0));
            dNs(0,14) = cPos(1)*cPos(2)*(2*cPos(0)+cPos(1)-2*cPos(2))-cPos(0)*cPos(1)*(cPos(0)+cPos(1)-4*cPos(2));
            dNs(1,14) = cPos(0)*cPos(2)*(cPos(0)+2*cPos(1)-2*cPos(2))-cPos(0)*cPos(1)*(cPos(0)+cPos(1)-4*cPos(2));
            dNs = cJac->invJ * dNs;
            break;
        default:
            throw std::runtime_error("shapeTria order not yet implemented");
        }
        break;
    default:
        throw std::runtime_error("shape cDim should be 2 or 3");
    }
}

shape::~shape()
{
    Ns.clear();
    dNs.clear();
    Nv.clear();
    dNv.clear();
}


jacobian::jacobian(size_t cDim, arma::mat cGeo)
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

jacobian::~jacobian()
{
    invJ.clear();
}


