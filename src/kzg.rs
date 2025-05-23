// This file includes backend tools:
// (1) read_line() is for taking inputs from the user
// (2) kzg_open_g1 is for opening KZG commitments
// (3) kzg_verify_g1 is for verifying KZG commitments
// (4) hash_caulk_single is for hashing group and field elements into a field
// element (5) random_field is for generating random field elements

use crate::{compute_h, compute_h_fake, field_dft, group_dft, util::convert_to_bigints};
use ark_ec::{msm::VariableBaseMSM, AffineCurve, PairingEngine, ProjectiveCurve};
use ark_ff::{Field, PrimeField};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial,
    UVPolynomial,
};
use ark_poly_commit::kzg10::*;
use ark_std::{end_timer, start_timer, One, Zero};
use std::marker::PhantomData;
use std::ops::DivAssign;
use std::time::Instant;
use ark_bn254::G2Affine;

/////////////////////////////////////////////////////////////////////
// KZG opening and verifying
/////////////////////////////////////////////////////////////////////

pub struct KZGCommit<E: PairingEngine> {
    phantom: PhantomData<E>,
}

impl<E: PairingEngine> KZGCommit<E> {
    pub fn commit_g1(powers: &Powers<E>, polynomial: &DensePolynomial<E::Fr>) -> E::G1Affine {
        let timer = start_timer!(|| "kzg g1 commit");
        let (com, _randomness) = KZG10::<E, _>::commit(powers, polynomial, None, None).unwrap();
        end_timer!(timer);
        com.0
    }

    pub fn commit_g2(g2_powers: &[E::G2Affine], poly: &DensePolynomial<E::Fr>) -> E::G2Affine {
        let timer = start_timer!(|| "kzg g2 commit");
        let poly_coeffs: Vec<<E::Fr as PrimeField>::BigInt> =
            poly.coeffs.iter().map(|&x| x.into_repr()).collect();
        let res = VariableBaseMSM::multi_scalar_mul(g2_powers, &poly_coeffs).into_affine();

        end_timer!(timer);
        res
    }

    // Function to commit to f(X,Y)
    // here f = [ [a0, a1, a2], [b1, b2, b3] ] represents (a0 + a1 Y + a2 Y^2 ) + X
    // (b1 + b2 Y + b3 Y^2)
    //
    // First we unwrap to get a vector of form [a0, a1, a2, b0, b1, b2]
    // Then we commit to f as a commitment to f'(X) = a0 + a1 X + a2 X^2 + b0 X^3 +
    // b1 X^4 + b2 X^5
    //
    // We also need to know the maximum degree of (a0 + a1 Y + a2 Y^2 ) to prevent
    // overflow errors.
    //
    // This is described in Section 4.6.2
    pub fn bipoly_commit(
        pp: &crate::multi::PublicParameters<E>,
        polys: &[DensePolynomial<E::Fr>],
        deg_x: usize,
    ) -> E::G1Affine {
        let timer = start_timer!(|| "kzg bipoly commit");
        let mut poly_formatted = Vec::new();

        for poly in polys {
            let temp = convert_to_bigints(&poly.coeffs);
            poly_formatted.extend_from_slice(&temp);
            for _ in poly.len()..deg_x {
                poly_formatted.push(E::Fr::zero().into_repr());
            }
        }

        assert!(pp.poly_ck.powers_of_g.len() >= poly_formatted.len());
        let g1_poly =
            VariableBaseMSM::multi_scalar_mul(&pp.poly_ck.powers_of_g, poly_formatted.as_slice())
                .into_affine();

        end_timer!(timer);
        g1_poly
    }

    // compute all openings to c_poly using a smart formula
    // This Code implements an algorithm for calculating n openings of a KZG vector
    // commitment of size n in n log(n) time. The algorithm is by Feist and
    // Khovratovich. It is useful for preprocessing.
    // The full algorithm is described here https://github.com/khovratovich/Kate/blob/master/Kate_amortized.pdf
    pub fn multiple_open<G>(
        c_poly: &DensePolynomial<E::Fr>, // c(X)
        powers: &[G],                    // SRS
        p: usize,
    ) -> Vec<G>
    where
        G: AffineCurve<ScalarField = E::Fr> + Sized,
    {
        let timer = start_timer!(|| "multiple open");

        let degree = c_poly.coeffs.len() - 1;
        let input_domain: GeneralEvaluationDomain<E::Fr> = EvaluationDomain::new(degree).unwrap();

        let h_timer = start_timer!(|| "compute h");
        let mut start = Instant::now();
        let powers: Vec<G::Projective> = powers.iter().map(|x| x.into_projective()).collect();
        let h2 = compute_h(c_poly, &powers, p);
        end_timer!(h_timer);
        println!("Computed h polynomial in time: {}s", start.elapsed().as_secs());

        let dom_size = input_domain.size();
        assert_eq!(1 << p, dom_size);
        assert_eq!(degree + 1, dom_size);

        let dft_timer = start_timer!(|| "G1 dft");
        start = Instant::now();
        let q2 = group_dft::<E::Fr, G::Projective>(&h2, p);
        end_timer!(dft_timer);
        println!("Computed DFT in time: {}s", start.elapsed().as_secs());

        let normalization_timer = start_timer!(|| "batch normalization");
        start = Instant::now();
        let res = G::Projective::batch_normalization_into_affine(q2.as_ref());
        end_timer!(normalization_timer);
        println!("Computed Batch Normalization in time: {}s", start.elapsed().as_secs());

        end_timer!(timer);
        res
    }

    pub fn multiple_open_fake<G>(
        c_poly: &DensePolynomial<E::Fr>, // c(X)
        powers: &[E::Fr],                    // SRS
        g1: G,
        p: usize,
    ) -> Vec<G>
        where
            G: AffineCurve<ScalarField = E::Fr> + Sized,
    {
        let timer = start_timer!(|| "multiple open");

        let degree = c_poly.coeffs.len() - 1;
        let input_domain: GeneralEvaluationDomain<E::Fr> = EvaluationDomain::new(degree).unwrap();

        let h_timer = start_timer!(|| "compute h");
        let mut start = Instant::now();
        //let powers: Vec<G::Projective> = powers.iter().map(|x| x.into_projective()).collect();
        let h2 = compute_h_fake(c_poly, &powers, p);
        end_timer!(h_timer);
        println!("Computed h polynomial in time: {}s", start.elapsed().as_secs());

        let dom_size = input_domain.size();
        assert_eq!(1 << p, dom_size);
        assert_eq!(degree + 1, dom_size);

        let dft_timer = start_timer!(|| "G1 dft");
        start = Instant::now();
        let q2 = field_dft::<E::Fr>(&h2, p);
        end_timer!(dft_timer);
        println!("Computed DFT in time: {}", start.elapsed().as_secs());

        let mut q3: Vec<G::Projective> = Vec::new();
        for i in 0..q2.len() {
            q3.push(g1.mul(q2[i]));
        }

        let normalization_timer = start_timer!(|| "batch normalization");
        start = Instant::now();
        let res = G::Projective::batch_normalization_into_affine(q3.as_ref());
        end_timer!(normalization_timer);
        println!("Computed Batch Normalization in time: {}s", start.elapsed().as_secs());

        end_timer!(timer);
        res
    }



    ////////////////////////////////////////////////
    // KZG.Open( srs_KZG, f(X, Y), deg, alpha )
    // returns ([f(alpha, x)]_1, pi)
    // Algorithm described in Section 4.6.2, KZG for Bivariate Polynomials
    pub fn partial_open_g1(
        pp: &crate::multi::PublicParameters<E>,
        polys: &[DensePolynomial<E::Fr>],
        deg_x: usize,
        point: &E::Fr,
    ) -> (E::G1Affine, E::G1Affine, DensePolynomial<E::Fr>) {
        let timer = start_timer!(|| "kzg partial open g1");
        let mut poly_partial_eval = DensePolynomial::from_coefficients_vec(vec![E::Fr::zero()]);
        let mut alpha = E::Fr::one();
        for coeff in polys {
            let pow_alpha = DensePolynomial::from_coefficients_vec(vec![alpha]);
            poly_partial_eval += &(&pow_alpha * coeff);
            alpha *= point;
        }

        let eval = VariableBaseMSM::multi_scalar_mul(
            &pp.poly_ck.powers_of_g,
            convert_to_bigints(&poly_partial_eval.coeffs).as_slice(),
        )
        .into_affine();

        let mut witness_bipolynomial = Vec::new();
        let poly_reverse: Vec<_> = polys.iter().rev().collect();
        witness_bipolynomial.push(poly_reverse[0].clone());

        let alpha = DensePolynomial::from_coefficients_vec(vec![*point]);
        for i in 1..(poly_reverse.len() - 1) {
            witness_bipolynomial.push(poly_reverse[i] + &(&alpha * &witness_bipolynomial[i - 1]));
        }

        witness_bipolynomial.reverse();

        let proof = Self::bipoly_commit(pp, &witness_bipolynomial, deg_x);

        end_timer!(timer);
        (eval, proof, poly_partial_eval)
    }

    // KZG.Open( srs_KZG, f(X), deg, (alpha1, alpha2, ..., alphan) )
    // returns ([f(alpha1), ..., f(alphan)], pi)
    // Algorithm described in Section 4.6.1, Multiple Openings
    pub fn open_g1_batch(
        poly_ck: &Powers<E>,
        poly: &DensePolynomial<E::Fr>,
        max_deg: Option<&usize>,
        points: &[E::Fr],
    ) -> (Vec<E::Fr>, E::G1Affine) {
        let timer = start_timer!(|| "kzg batch open g1");
        let mut evals = Vec::new();
        let mut proofs = Vec::new();
        for p in points.iter() {
            let (eval, pi) = Self::open_g1_single(poly_ck, poly, max_deg, p);
            evals.push(eval);
            proofs.push(pi);
        }

        let mut res = E::G1Projective::zero(); // default value

        for j in 0..points.len() {
            let w_j = points[j];
            // 1. Computing coefficient [1/prod]
            let mut prod = E::Fr::one();
            for (k, p) in points.iter().enumerate() {
                if k != j {
                    prod *= w_j - p;
                }
            }
            // 2. Summation
            let q_add = proofs[j].mul(prod.inverse().unwrap()); //[1/prod]Q_{j}
            res += q_add;
        }

        end_timer!(timer);
        (evals, res.into_affine())
    }

    // KZG.Open( srs_KZG, f(X), deg, alpha ) returns (f(alpha), pi)
    fn open_g1_single(
        poly_ck: &Powers<E>,
        poly: &DensePolynomial<E::Fr>,
        max_deg: Option<&usize>,
        point: &E::Fr,
    ) -> (E::Fr, E::G1Affine) {
        let timer = start_timer!(|| "kzg open g1");
        let eval = poly.evaluate(point);

        let global_max_deg = poly_ck.powers_of_g.len();

        let mut d: usize = 0;
        if max_deg == None {
            d += global_max_deg;
        } else {
            d += max_deg.unwrap();
        }
        let divisor = DensePolynomial::from_coefficients_vec(vec![-*point, E::Fr::one()]);
        let witness_polynomial = poly / &divisor;

        assert!(poly_ck.powers_of_g[(global_max_deg - d)..].len() >= witness_polynomial.len());
        let proof = VariableBaseMSM::multi_scalar_mul(
            &poly_ck.powers_of_g[(global_max_deg - d)..],
            convert_to_bigints(&witness_polynomial.coeffs).as_slice(),
        )
        .into_affine();

        end_timer!(timer);
        (eval, proof)
    }

    pub fn verify_proof_g1(
        powers_of_g1: &[E::G1Affine],   // generators of G1
        powers_of_g2: &[E::G2Affine],   // generators of G2
        c_com: &E::G1Affine,
        point: &E::Fr,
        eval: &E::Fr,
        pi: &E::G1Affine,
    ) -> bool {
        let mut status: bool  = false;
        let g1 = powers_of_g1[0];
        let g1x = powers_of_g1[1];
        let g2 = powers_of_g2[0];
        let g2x = powers_of_g2[1];
        let ptneg = -*point;

        let rhs:E::G2Affine = g2x + g2.mul(-*point).into_affine();
        let lhs:E::G1Affine = *c_com + g1.mul(-*eval).into_affine();
        let mut prod = E::pairing(g1x, g2);
        prod.div_assign(&E::pairing(g1, g2x));
        //println!("lp: {:?}", lp);
        //println!("rp: {:?}", rp);
        status = (prod.is_one());
        status
    }

    // KZG.Verify( srs_KZG, F, deg, (alpha1, alpha2, ..., alphan), (v1, ..., vn), pi
    // ) Algorithm described in Section 4.6.1, Multiple Openings
    pub fn verify_g1(
        // Verify that @c_com is a commitment to C(X) such that C(x)=z
        powers_of_g1: &[E::G1Affine], // generator of G1
        powers_of_g2: &[E::G2Affine], // [1]_2, [x]_2, [x^2]_2, ...
        c_com: &E::G1Affine,          // commitment
        max_deg: Option<&usize>,      // max degree
        points: &[E::Fr],             // x such that eval = C(x)
        evals: &[E::Fr],              // evaluation
        pi: &E::G1Affine,             // proof
    ) -> bool {
        let timer = start_timer!(|| "kzg verify g1");
        let pairing_inputs = Self::verify_g1_defer_pairing(
            powers_of_g1,
            powers_of_g2,
            c_com,
            max_deg,
            points,
            evals,
            pi,
        );

        let pairing_timer = start_timer!(|| "pairing product");
        let prepared_pairing_inputs = vec![
            (
                E::G1Prepared::from(pairing_inputs[0].0.into_affine()),
                E::G2Prepared::from(pairing_inputs[0].1.into_affine()),
            ),
            (
                E::G1Prepared::from(pairing_inputs[1].0.into_affine()),
                E::G2Prepared::from(pairing_inputs[1].1.into_affine()),
            ),
        ];
        let res = E::product_of_pairings(prepared_pairing_inputs.iter()).is_one();

        end_timer!(pairing_timer);
        end_timer!(timer);
        res
    }

    // KZG.Verify( srs_KZG, F, deg, (alpha1, alpha2, ..., alphan), (v1, ..., vn), pi
    // ) Algorithm described in Section 4.6.1, Multiple Openings
    pub fn verify_g1_defer_pairing(
        // Verify that @c_com is a commitment to C(X) such that C(x)=z
        powers_of_g1: &[E::G1Affine], // generator of G1
        powers_of_g2: &[E::G2Affine], // [1]_2, [x]_2, [x^2]_2, ...
        c_com: &E::G1Affine,          // commitment
        max_deg: Option<&usize>,      // max degree
        points: &[E::Fr],             // x such that eval = C(x)
        evals: &[E::Fr],              // evaluation
        pi: &E::G1Affine,             // proof
    ) -> Vec<(E::G1Projective, E::G2Projective)> {
        let timer = start_timer!(|| "kzg verify g1 (deferring pairing)");

        // Interpolation set
        // tau_i(X) = lagrange_tau[i] = polynomial equal to 0 at point[j] for j!= i and
        // 1  at points[i]

        let mut lagrange_tau = DensePolynomial::from_coefficients_slice(&[E::Fr::zero()]);
        let mut prod = DensePolynomial::from_coefficients_slice(&[E::Fr::one()]);
        let mut components = vec![];
        for &p in points.iter() {
            let poly = DensePolynomial::from_coefficients_slice(&[-p, E::Fr::one()]);
            prod = &prod * (&poly);
            components.push(poly);
        }

        for i in 0..points.len() {
            let mut temp = &prod / &components[i];
            let lagrange_scalar = temp.evaluate(&points[i]).inverse().unwrap() * evals[i];
            temp.coeffs.iter_mut().for_each(|x| *x *= lagrange_scalar);
            lagrange_tau = lagrange_tau + temp;
        }

        // commit to sum evals[i] tau_i(X)
        assert!(
            powers_of_g1.len() >= lagrange_tau.len(),
            "KZG verifier doesn't have enough g1 powers"
        );
        let g1_tau = VariableBaseMSM::multi_scalar_mul(
            &powers_of_g1[..lagrange_tau.len()],
            convert_to_bigints(&lagrange_tau.coeffs).as_slice(),
        );

        // vanishing polynomial
        let z_tau = prod;

        // commit to z_tau(X) in g2
        assert!(
            powers_of_g2.len() >= z_tau.len(),
            "KZG verifier doesn't have enough g2 powers"
        );
        let g2_z_tau = VariableBaseMSM::multi_scalar_mul(
            &powers_of_g2[..z_tau.len()],
            convert_to_bigints(&z_tau.coeffs).as_slice(),
        );

        let global_max_deg = powers_of_g1.len();

        let mut d: usize = 0;
        if max_deg == None {
            d += global_max_deg;
        } else {
            d += max_deg.unwrap();
        }

        let res = vec![
            (
                g1_tau - c_com.into_projective(),
                powers_of_g2[global_max_deg - d].into_projective(),
            ),
            (pi.into_projective(), g2_z_tau),
        ];

        end_timer!(timer);
        res
    }

    // KZG.Verify( srs_KZG, F, deg, alpha, F_alpha, pi )
    // Algorithm described in Section 4.6.2, KZG for Bivariate Polynomials
    // Be very careful here.  Verification is only valid if it is paired with a
    // degree check.
    pub fn partial_verify_g1(
        srs: &crate::multi::PublicParameters<E>,
        c_com: &E::G1Affine, // commitment
        deg_x: usize,
        point: &E::Fr,
        partial_eval: &E::G1Affine,
        pi: &E::G1Affine, // proof
    ) -> bool {
        let timer = start_timer!(|| "kzg partial verify g1");
        let pairing_inputs =
            Self::partial_verify_g1_defer_pairing(srs, c_com, deg_x, point, partial_eval, pi);
        let pairing_timer = start_timer!(|| "pairing product");
        let prepared_pairing_inputs = vec![
            (
                E::G1Prepared::from(pairing_inputs[0].0.into_affine()),
                E::G2Prepared::from(pairing_inputs[0].1.into_affine()),
            ),
            (
                E::G1Prepared::from(pairing_inputs[1].0.into_affine()),
                E::G2Prepared::from(pairing_inputs[1].1.into_affine()),
            ),
        ];

        let res = E::product_of_pairings(prepared_pairing_inputs.iter()).is_one();

        end_timer!(pairing_timer);
        end_timer!(timer);
        res
    }

    // KZG.Verify( srs_KZG, F, deg, alpha, F_alpha, pi )
    // Algorithm described in Section 4.6.2, KZG for Bivariate Polynomials
    // Be very careful here.  Verification is only valid if it is paired with a
    // degree check.
    pub fn partial_verify_g1_defer_pairing(
        srs: &crate::multi::PublicParameters<E>,
        c_com: &E::G1Affine, // commitment
        deg_x: usize,
        point: &E::Fr,
        partial_eval: &E::G1Affine,
        pi: &E::G1Affine, // proof
    ) -> Vec<(E::G1Projective, E::G2Projective)> {
        let timer = start_timer!(|| "kzg partial verify g1 (deferring pairing)");
        let res = vec![
            (
                partial_eval.into_projective() - c_com.into_projective(),
                srs.g2_powers[0].into_projective(),
            ),
            (
                pi.into_projective(),
                srs.g2_powers[deg_x].into_projective() - srs.g2_powers[0].mul(*point),
            ),
        ];
        end_timer!(timer);
        res
    }

    // Algorithm for aggregating KZG proofs into a single proof
    // Described in Section 4.6.3 Subset openings
    // compute Q =\sum_{j=1}^m \frac{Q_{i_j}}}{\prod_{1\leq k\leq m,\; k\neq
    // j}(\omega^{i_j}-\omega^{i_k})}
    pub fn aggregate_proof_g2(
        openings: &[E::G2Affine], // Q_i
        positions: &[usize],      // i_j
        input_domain: &GeneralEvaluationDomain<E::Fr>,
    ) -> E::G2Affine {
        let timer = start_timer!(|| "kzg aggregate proof");

        let m = positions.len();
        let mut res = openings[0].into_projective(); // default value

        for j in 0..m {
            let i_j = positions[j];
            let w_ij = input_domain.element(i_j);
            // 1. Computing coefficient [1/prod]
            let mut prod = E::Fr::one();
            for (k, &pos) in positions.iter().enumerate().take(m) {
                let w_ik = input_domain.element(pos);
                if k != j {
                    prod *= w_ij - w_ik;
                }
            }
            // 2. Summation
            let q_add = openings[i_j].mul(prod.inverse().unwrap()); //[1/prod]Q_{j}
            if j == 0 {
                res = q_add;
            } else {
                res += q_add;
            }
        }
        let res = res.into_affine();
        end_timer!(timer);
        res
    }
}

pub fn generate_lagrange_polynomials_subset<E: PairingEngine>(
    positions: &[usize],
    srs: &crate::multi::PublicParameters<E>,
) -> Vec<DensePolynomial<E::Fr>> {
    let timer = start_timer!(|| "generate lagrange poly subset");

    let mut tau_polys = vec![];
    let m = positions.len();
    for j in 0..m {
        let mut tau_j = DensePolynomial::from_coefficients_slice(&[E::Fr::one()]); // start from tau_j =1
        for k in 0..m {
            if k != j {
                // tau_j = prod_{k\neq j} (X-w^(i_k))/(w^(i_j)-w^(i_k))
                let denum = srs.domain_N.element(positions[j]) - srs.domain_N.element(positions[k]);
                let denum = E::Fr::one() / denum;
                tau_j = &tau_j
                    * &DensePolynomial::from_coefficients_slice(&[
                        -srs.domain_N.element(positions[k]) * denum.clone(), //-w^(i_k))/(w^(i_j)-w^(i_k)
                        denum,                                       // 1//(w^(i_j)-w^(i_k))
                    ]);
            }
        }
        tau_polys.push(tau_j.clone());
    }
    end_timer!(timer);
    tau_polys
}

#[cfg(test)]
pub mod tests {

    use super::{generate_lagrange_polynomials_subset, KZGCommit, *};
    use crate::caulk_single_setup;
    use ark_bls12_377::Bls12_377;
    use ark_bls12_381::Bls12_381;
    use ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
    use ark_poly::{univariate::DensePolynomial, EvaluationDomain, Polynomial, UVPolynomial};
    use ark_poly_commit::kzg10::KZG10;
    use ark_std::{test_rng, One, Zero};
    use std::time::Instant;

    #[test]
    fn test_lagrange() {
        test_lagrange_helper::<Bls12_377>();
        test_lagrange_helper::<Bls12_381>();
    }

    #[allow(non_snake_case)]
    fn test_lagrange_helper<E: PairingEngine>() {
        let p: usize = 8; // bitlength of poly degree
        let max_degree: usize = (1 << p) + 2;
        let m: usize = 8;
        let N: usize = 1 << p;

        let now = Instant::now();
        let pp = crate::multi::PublicParameters::<E>::setup(&max_degree, &N, &m, &p, false);
        println!("time to setup {:?}", now.elapsed());

        let mut positions: Vec<usize> = vec![];
        for i in 0..m {
            // generate positions evenly distributed in the set
            let i_j: usize = i * (max_degree / m);
            positions.push(i_j);
        }

        let tau_polys = generate_lagrange_polynomials_subset(&positions, &pp);
        for j in 0..m {
            for k in 0..m {
                if k == j {
                    assert_eq!(
                        tau_polys[j].evaluate(&pp.domain_N.element(positions[k])),
                        E::Fr::one()
                    )
                } else {
                    assert_eq!(
                        tau_polys[j].evaluate(&pp.domain_N.element(positions[k])),
                        E::Fr::zero()
                    )
                }
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    pub fn test_Q_g2() {
        test_Q_g2_helper::<Bls12_381>();
        test_Q_g2_helper::<Bls12_377>();
    }

    #[allow(non_snake_case)]
    pub fn test_Q_g2_helper<E: PairingEngine>() {
        let rng = &mut ark_std::test_rng();

        // current kzg setup should be changed with output from a setup ceremony
        let p: usize = 6; // bitlength of poly degree
        let max_degree: usize = (1 << p) + 2;
        let actual_degree: usize = (1 << p) - 1;
        let m: usize = 1 << (p / 2);
        let N: usize = 1 << p;
        let pp = crate::multi::PublicParameters::setup(&max_degree, &N, &m, &p, false);

        // Setting up test instance to run evaluate on.
        // test randomness for c_poly is same everytime.
        // test index equals 5 everytime
        // g_c = g^(c(x))

        let c_poly = DensePolynomial::<E::Fr>::rand(actual_degree, rng);
        let c_com = KZGCommit::<E>::commit_g1(&pp.poly_ck, &c_poly);

        let now = Instant::now();
        let openings = KZGCommit::<E>::multiple_open::<E::G2Affine>(&c_poly, &pp.g2_powers, p);
        println!("Multi advanced computed in {:?}", now.elapsed());

        let mut positions: Vec<usize> = vec![];
        for i in 0..m {
            let i_j: usize = i * (max_degree / m);
            positions.push(i_j);
        }

        let now = Instant::now();

        // Compute proof
        let Q: E::G2Affine =
            KZGCommit::<E>::aggregate_proof_g2(&openings, &positions, &pp.domain_N);
        println!(
            "Full proof for {:?} positions computed in {:?}",
            m,
            now.elapsed()
        );

        // Compute commitment to C_I
        let mut C_I = DensePolynomial::from_coefficients_slice(&[E::Fr::zero()]); // C_I =  sum_j c_j*tau_j
        let tau_polys = generate_lagrange_polynomials_subset(&positions, &pp);
        for j in 0..m {
            C_I = &C_I + &(&tau_polys[j] * c_poly.evaluate(&pp.domain_N.element(positions[j])));
            // sum_j c_j*tau_j
        }
        let c_I_com = KZGCommit::<E>::commit_g1(&pp.poly_ck, &C_I);

        // Compute commitment to z_I
        let mut z_I = DensePolynomial::from_coefficients_slice(&[E::Fr::one()]);
        for j in 0..m {
            z_I = &z_I
                * &DensePolynomial::from_coefficients_slice(&[
                    -pp.domain_N.element(positions[j]),
                    E::Fr::one(),
                ]);
        }
        let z_I_com = KZGCommit::<E>::commit_g1(&pp.poly_ck, &z_I);

        // pairing check
        let pairing1 = E::pairing(
            (c_com.into_projective() - c_I_com.into_projective()).into_affine(),
            pp.g2_powers[0],
        );
        let pairing2 = E::pairing(z_I_com, Q);
        assert_eq!(pairing1, pairing2);
    }

    #[test]
    fn test_single() {
        test_single_helper::<Bls12_381>();
        test_single_helper::<Bls12_377>();
    }

    fn test_single_helper<E: PairingEngine>() {
        let mut rng = test_rng();

        // setting public parameters
        // current kzg setup should be changed with output from a setup ceremony
        let max_degree: usize = 100;
        let actual_degree: usize = 63;
        let pp = caulk_single_setup(max_degree, actual_degree, &mut rng);

        // Setting up test instance to run evaluate on.
        // test randomness for c_poly is same everytime.
        // test index equals 5 everytime
        // g_c = g^(c(x))
        let rng = &mut test_rng();
        let c_poly = DensePolynomial::<E::Fr>::rand(actual_degree, rng);
        let (_c_com, c_com_open) = KZG10::<E, _>::commit(&pp.poly_ck, &c_poly, None, None).unwrap();

        let i: usize = 6;
        let q = single_open_default(&c_poly, &c_com_open, &pp.poly_ck, i, actual_degree);
        let q2 = single_open_fast(&c_poly, &pp.poly_ck, i, actual_degree);
        assert_eq!(q, q2);
    }

    #[test]
    pub fn test_multi() {
        test_multi_helper::<Bls12_381>();
        test_multi_helper::<Bls12_377>();
    }

    pub fn test_multi_helper<E: PairingEngine>() {
        let mut rng = test_rng();

        // current kzg setup should be changed with output from a setup ceremony
        let p: usize = 9;
        let max_degree: usize = 1 << p + 1;
        let actual_degree: usize = (1 << p) - 1;
        let pp = caulk_single_setup(max_degree, actual_degree, &mut rng);

        // Setting up test instance to run evaluate on.
        // test randomness for c_poly is same everytime.
        // test index equals 5 everytime
        // g_c = g^(c(x))
        let c_poly = DensePolynomial::<E::Fr>::rand(actual_degree, &mut rng);
        let (c_com, c_com_open) = KZG10::<E, _>::commit(&pp.poly_ck, &c_poly, None, None).unwrap();
        let _g_c = c_com.0;

        let now = Instant::now();
        let q = multiple_open_naive(&c_poly, &c_com_open, &pp.poly_ck, actual_degree);
        println!("Multi naive computed in {:?}", now.elapsed());

        let now = Instant::now();
        let q2 = KZGCommit::<E>::multiple_open::<E::G1Affine>(&c_poly, &pp.poly_ck.powers_of_g, p);
        println!("Multi advanced computed in {:?}", now.elapsed());
        assert_eq!(q, q2);
    }

    #[test]
    fn test_commit() {
        test_commit_helper::<Bls12_381>();
        test_commit_helper::<Bls12_377>();
    }

    pub fn test_commit_helper<E: PairingEngine>() {
        let mut rng = test_rng();

        // current kzg setup should be changed with output from a setup ceremony
        let max_degree: usize = 100;
        let actual_degree: usize = 63;
        let pp = caulk_single_setup(max_degree, actual_degree, &mut rng);

        // Setting up test instance to run evaluate on.
        // test randomness for c_poly is same everytime.
        // g_c = g^(c(x))
        let c_poly = DensePolynomial::<E::Fr>::rand(actual_degree, &mut rng);
        let (c_com, _c_com_open) = KZG10::<E, _>::commit(&pp.poly_ck, &c_poly, None, None).unwrap();
        let g_c1 = c_com.0;

        let g_c2 = commit_direct(&c_poly, &pp.poly_ck);
        assert_eq!(g_c1, g_c2);
        println!("commit test passed")
    }

    /// Various functions that are used for testing

    fn commit_direct<E: PairingEngine>(
        c_poly: &DensePolynomial<E::Fr>, // c(X)
        poly_ck: &Powers<E>,             // SRS
    ) -> E::G1Affine {
        assert!(c_poly.coeffs.len() <= poly_ck.powers_of_g.len());
        let mut com = poly_ck.powers_of_g[0].mul(c_poly.coeffs[0]);
        for i in 1..c_poly.coeffs.len() {
            com = com + poly_ck.powers_of_g[i].mul(c_poly.coeffs[i]);
        }
        com.into_affine()
    }

    // compute all openings to c_poly by mere calling `open` N times
    fn multiple_open_naive<E: PairingEngine>(
        c_poly: &DensePolynomial<E::Fr>,
        c_com_open: &Randomness<E::Fr, DensePolynomial<E::Fr>>,
        poly_ck: &Powers<E>,
        degree: usize,
    ) -> Vec<E::G1Affine> {
        let input_domain: GeneralEvaluationDomain<E::Fr> = EvaluationDomain::new(degree).unwrap();
        let mut res: Vec<E::G1Affine> = vec![];
        for i in 0..input_domain.size() {
            let omega_i = input_domain.element(i);
            res.push(kzg_open_g1_test::<E>(&c_poly, &omega_i, &c_com_open, &poly_ck).w);
        }
        res
    }

    ////////////////////////////////////////////////
    fn kzg_open_g1_test<E: PairingEngine>(
        p: &DensePolynomial<E::Fr>,
        omega_5: &E::Fr,
        polycom_open: &Randomness<E::Fr, DensePolynomial<E::Fr>>,
        poly_ck: &Powers<E>,
    ) -> Proof<E> {
        let rng = &mut ark_std::test_rng();

        let (witness_polynomial, _random_witness_polynomial) =
            KZG10::<E, _>::compute_witness_polynomial(p, omega_5.clone(), polycom_open).unwrap();

        let (temp0, _temp1) = KZG10::commit(poly_ck, &witness_polynomial, None, Some(rng)).unwrap();
        Proof {
            w: temp0.0,
            random_v: None,
        }
    }

    // compute KZG proof   Q = g1_q = g^( (c(x) - c(w^i)) / (x - w^i) ) where x is
    // secret, w^i is the point where we open, and c(X) is the committed  polynomial
    fn single_open_default<E: PairingEngine>(
        c_poly: &DensePolynomial<E::Fr>,                        // c(X)
        c_com_open: &Randomness<E::Fr, DensePolynomial<E::Fr>>, //
        poly_ck: &Powers<E>,
        i: usize, //
        degree: usize,
    ) -> E::G1Affine {
        let input_domain: GeneralEvaluationDomain<E::Fr> = EvaluationDomain::new(degree).unwrap();
        let omega_i = input_domain.element(i);
        let c_poly_open = kzg_open_g1_test(&c_poly, &omega_i, &c_com_open, &poly_ck);
        c_poly_open.w
    }

    // KZG proof/opening at point y for c(X) = sum_i c_i X^i
    //(1)T_y(X) = sum_i t_i X^i
    //(2) t_{deg-1} = c_deg
    //(3) t_j = c_{j+1} + y*t_{j+1}
    fn single_open_fast<E: PairingEngine>(
        c_poly: &DensePolynomial<E::Fr>, // c(X)
        poly_ck: &Powers<E>,             // SRS
        i: usize,                        // y=w^i
        degree: usize,                   // degree of c(X)
    ) -> E::G1Affine {
        // computing opening point
        let input_domain: GeneralEvaluationDomain<E::Fr> = EvaluationDomain::new(degree).unwrap();
        let y = input_domain.element(i);

        // compute quotient
        let mut t_poly = c_poly.clone();
        t_poly.coeffs.remove(0); // shifting indices
        for j in (0..t_poly.len() - 1).rev() {
            t_poly.coeffs[j] = c_poly.coeffs[j + 1] + y * t_poly.coeffs[j + 1]
        }

        // commit
        let (t_com, _) = KZG10::commit(&poly_ck, &t_poly, None, None).unwrap();
        t_com.0
    }
}
