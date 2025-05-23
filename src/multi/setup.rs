use crate::util::trim;
use ark_bls12_377::g2;
use ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
use ark_ff::{PrimeField, UniformRand};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    GeneralEvaluationDomain, UVPolynomial,
};
use ark_poly_commit::kzg10::{UniversalParams, Powers, VerifierKey, KZG10};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, One, Zero};
use ark_test_curves::pairing::Pairing;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    convert::TryInto,
    fs::File,
    io::{Read, Write},
    time::Instant,
};
use ark_bls12_381::Bls12_381;
use ark_bn254::Bn254;
use ark_std::rand::RngCore;

// my dummy kzg10 setup

pub fn kzg10_setup_dummy<E: PairingEngine, R:RngCore>(
    max_degree: usize,
    tmp_g: E::G1Affine,
    rng: &mut R,
) -> UniversalParams<E> {
    let g_rand = E::G1Projective::rand(rng);
    let h_rand = E::G2Projective::rand(rng);
    // let f_rand = E::Fr::rand(rng);
    let g_rand_affine = g_rand.into_affine();
    let h_rand_affine = h_rand.into_affine();
    UniversalParams{
        powers_of_g: vec![g_rand_affine; max_degree + 1],
        powers_of_gamma_g: (vec![g_rand_affine; max_degree + 1]).into_iter().enumerate().collect(),
        h:h_rand_affine,
        beta_h:h_rand_affine,
        neg_powers_of_h: vec![h_rand_affine; max_degree + 1].into_iter().enumerate().collect(),
        prepared_h: E::G2Prepared::from(h_rand.into_affine()),
        prepared_beta_h: E::G2Prepared::from(h_rand.into_affine()),
    }
}

// structure of public parameters
#[allow(non_snake_case)]
pub struct PublicParameters<E: PairingEngine> {
    pub poly_ck: Powers<'static, E>,
    pub domain_m: GeneralEvaluationDomain<E::Fr>,
    pub domain_n: GeneralEvaluationDomain<E::Fr>,
    pub domain_N: GeneralEvaluationDomain<E::Fr>,
    pub verifier_pp: VerifierPublicParameters<E>,
    pub lagrange_polynomials_n: Vec<DensePolynomial<E::Fr>>,
    pub lagrange_polynomials_m: Vec<DensePolynomial<E::Fr>>,
    pub id_poly: DensePolynomial<E::Fr>,
    pub N: usize,
    pub m: usize,
    pub n: usize,
    pub g2_powers: Vec<E::G2Affine>,
}

pub struct LookupParameters<F: PrimeField> {
    m: usize,
    lagrange_polynomials_m: Vec<DensePolynomial<F>>,
    domain_m: GeneralEvaluationDomain<F>,
    id_poly: DensePolynomial<F>,
}

impl<F: PrimeField> LookupParameters<F> {
    fn new(m: usize) -> Self {
        let domain_m: GeneralEvaluationDomain<F> = GeneralEvaluationDomain::new(m).unwrap();

        // id_poly(X) = 1 for omega_m in range and 0 for omega_m not in range.
        let mut id_vec = Vec::new();
        for _ in 0..m {
            id_vec.push(F::one());
        }
        for _ in m..domain_m.size() {
            id_vec.push(F::zero());
        }
        let id_poly = EvaluationsOnDomain::from_vec_and_domain(id_vec, domain_m).interpolate();
        let mut lagrange_polynomials_m: Vec<DensePolynomial<F>> = Vec::new();

        for i in 0..domain_m.size() {
            let evals: Vec<F> = cfg_into_iter!(0..domain_m.size())
                .map(|k| if k == i { F::one() } else { F::zero() })
                .collect();
            lagrange_polynomials_m
                .push(EvaluationsOnDomain::from_vec_and_domain(evals, domain_m).interpolate());
        }

        Self {
            m,
            lagrange_polynomials_m,
            domain_m,
            id_poly,
        }
    }
}

// smaller set of public parameters used by verifier
pub struct VerifierPublicParameters<E: PairingEngine> {
    pub poly_vk: VerifierKey<E>,
    pub domain_m_size: usize,
}

// pub fn dummy_kzg10_setup<E:PairingEngine>(
//     max_degree: usize,
//     fr: E::Fr,
//     g1r: E::G1Affine,
//     g2r: E::G2Affine,
// ) -> ark_poly_commit::kzg10::UniversalParams<E> {
//     UniversalParams { powers_of_g: vec![g1r; max_degree+1], powers_of_gamma_g: vec![g1r; max_degree+1], h: g2r, beta_h: g2r, neg_powers_of_h: vec![g2r;max_degree+1], prepared_h: g2r.into(), prepared_beta_h: g2r.into()}
// }

impl<E: PairingEngine> PublicParameters<E> {
    pub fn regenerate_lookup_params(&mut self, m: usize) {
        let lp = LookupParameters::new(m);
        self.m = lp.m;
        self.lagrange_polynomials_m = lp.lagrange_polynomials_m;
        self.domain_m = lp.domain_m;
        self.id_poly = lp.id_poly;
    }

    // store powers of g in a file
    pub fn store(&self, path: &str) {
        // 1. Powers of g
        let mut g_bytes = vec![];
        let mut f = File::create(path).expect("Unable to create file");
        let deg: u32 = self.poly_ck.powers_of_g.len().try_into().unwrap();
        let deg_bytes = deg.to_be_bytes();
        f.write_all(&deg_bytes).expect("Unable to write data");
        let deg32: usize = deg.try_into().unwrap();
        for i in 0..deg32 {
            self.poly_ck.powers_of_g[i]
                .into_projective()
                .into_affine()
                .serialize_uncompressed(&mut g_bytes)
                .ok();
        }
        f.write_all(&g_bytes).expect("Unable to write data");

        // 2. Powers of gammag
        let deg_gamma: u32 = self.poly_ck.powers_of_gamma_g.len().try_into().unwrap();
        let mut gg_bytes = vec![];
        let deg_bytes = deg_gamma.to_be_bytes();
        f.write_all(&deg_bytes).expect("Unable to write data");
        let deg32: usize = deg.try_into().unwrap();
        for i in 0..deg32 {
            self.poly_ck.powers_of_gamma_g[i]
                .into_projective()
                .into_affine()
                .serialize_uncompressed(&mut gg_bytes)
                .ok();
        }
        f.write_all(&gg_bytes).expect("Unable to write data");

        // 3. Verifier key
        let mut h_bytes = vec![];
        self.verifier_pp
            .poly_vk
            .h
            .serialize_uncompressed(&mut h_bytes)
            .ok();
        self.verifier_pp
            .poly_vk
            .beta_h
            .serialize_uncompressed(&mut h_bytes)
            .ok();
        f.write_all(&h_bytes).expect("Unable to write data");

        // 4. g2 powers
        let mut g2_bytes = vec![];
        let deg2: u32 = self.g2_powers.len().try_into().unwrap();
        let deg2_bytes = deg2.to_be_bytes();
        f.write_all(&deg2_bytes).expect("Unable to write data");
        let deg2_32: usize = deg2.try_into().unwrap();
        for i in 0..deg2_32 {
            self.g2_powers[i]
                .into_projective()
                .into_affine()
                .serialize_uncompressed(&mut g2_bytes)
                .ok();
        }
        f.write_all(&g2_bytes).expect("Unable to write data");
    }

    // load powers of g from a file
    pub fn load(path: &str) -> (Powers<'static, E>, VerifierKey<E>, Vec<E::G2Affine>) {
        const G1_UNCOMPR_SIZE: usize = 96;
        const G2_UNCOMPR_SIZE: usize = 192;
        let mut data = Vec::new();
        let mut f = File::open(path).expect("Unable to open file");
        f.read_to_end(&mut data).expect("Unable to read data");

        // 1. reading g powers
        let mut cur_counter: usize = 0;
        let deg_bytes: [u8; 4] = (&data[0..4]).try_into().unwrap();
        let deg: u32 = u32::from_be_bytes(deg_bytes);
        let mut powers_of_g = vec![];
        let deg32: usize = deg.try_into().unwrap();
        cur_counter += 4;
        for i in 0..deg32 {
            let buf_bytes =
                &data[cur_counter + i * G1_UNCOMPR_SIZE..cur_counter + (i + 1) * G1_UNCOMPR_SIZE];
            let tmp = E::G1Affine::deserialize_unchecked(buf_bytes).unwrap();
            powers_of_g.push(tmp);
        }
        cur_counter += deg32 * G1_UNCOMPR_SIZE;

        // 2. reading gamma g powers
        let deg_bytes: [u8; 4] = (&data[cur_counter..cur_counter + 4]).try_into().unwrap();
        let deg: u32 = u32::from_be_bytes(deg_bytes);
        let mut powers_of_gamma_g = vec![];
        let deg32: usize = deg.try_into().unwrap();
        cur_counter += 4;
        for i in 0..deg32 {
            let buf_bytes =
                &data[cur_counter + i * G1_UNCOMPR_SIZE..cur_counter + (i + 1) * G1_UNCOMPR_SIZE];
            let tmp = E::G1Affine::deserialize_unchecked(buf_bytes).unwrap();
            powers_of_gamma_g.push(tmp);
        }
        cur_counter += deg32 * G1_UNCOMPR_SIZE;

        // 3. reading verifier key
        let buf_bytes = &data[cur_counter..cur_counter + G2_UNCOMPR_SIZE];
        let h = E::G2Affine::deserialize_unchecked(buf_bytes).unwrap();
        cur_counter += G2_UNCOMPR_SIZE;
        let buf_bytes = &data[cur_counter..cur_counter + G2_UNCOMPR_SIZE];
        let beta_h = E::G2Affine::deserialize_unchecked(buf_bytes).unwrap();
        cur_counter += G2_UNCOMPR_SIZE;

        // 4. reading G2 powers
        let deg2_bytes: [u8; 4] = (&data[cur_counter..cur_counter + 4]).try_into().unwrap();
        let deg2: u32 = u32::from_be_bytes(deg2_bytes);
        let mut g2_powers = vec![];
        let deg2_32: usize = deg2.try_into().unwrap();
        cur_counter += 4;
        for _ in 0..deg2_32 {
            let buf_bytes = &data[cur_counter..cur_counter + G2_UNCOMPR_SIZE];
            let tmp = E::G2Affine::deserialize_unchecked(buf_bytes).unwrap();
            g2_powers.push(tmp);
            cur_counter += G2_UNCOMPR_SIZE;
        }

        let vk = VerifierKey {
            g: powers_of_g[0],
            gamma_g: powers_of_gamma_g[0],
            h,
            beta_h,
            prepared_h: h.into(),
            prepared_beta_h: beta_h.into(),
        };

        let powers = Powers {
            powers_of_g: ark_std::borrow::Cow::Owned(powers_of_g),
            powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
        };

        (powers, vk, g2_powers)
    }

    // setup algorithm for index_hiding_polycommit
    // also includes a bunch of precomputation.
    // @max_degree max degree of table polynomial C(X), also the size of the trusted
    // setup @N domain size on which proofs are constructed. Should not be
    // smaller than max_degree @m lookup size. Can be c hanged later
    // @n suppl domain for the unity proofs. Should be at least 6+log N
    #[allow(non_snake_case)]
    pub fn setup(max_degree: &usize, N: &usize, m: &usize, n: &usize, dummy:bool) -> PublicParameters<E> {
        // Setup algorithm. To be replaced by output of a universal setup before being
        // production ready.

        // let mut srs = KzgBls12_381::setup(4, true, rng).unwrap();
        let poly_ck: Powers<'static, E>;
        let poly_vk: VerifierKey<E>;
        let mut g2_powers: Vec<E::G2Affine> = Vec::new();

        // try opening the file. If it exists load the setup from there, otherwise
        // generate
        let path = format!("srs/srs_{}_{}.setup", max_degree, E::Fq::size_in_bits());
        let mut res = File::open(path.clone());
        let store_to_file: bool;
        res = Err(std::io::Error::new(std::io::ErrorKind::NotFound, "dummy"));
        match res {
            Ok(_) => {
                let now = Instant::now();
                let (_poly_ck, _poly_vk, _g2_powers) = PublicParameters::load(&path);
                println!("time to load powers = {:?}", now.elapsed());
                store_to_file = false;
                g2_powers = _g2_powers;
                poly_ck = _poly_ck;
                poly_vk = _poly_vk;
            },
            Err(_) => {
                let rng = &mut ark_std::test_rng();
                let tmp_g = E::G1Projective::rand(rng).into_affine();
                let now = Instant::now();
                let srs = match dummy{
                    true => kzg10_setup_dummy(*max_degree, tmp_g, rng), 
                    false => KZG10::<E, DensePolynomial<E::Fr>>::setup(*max_degree, true, rng).unwrap()};
                println!("time to setup powers = {:?}", now.elapsed());

                // trim down to size
                let (poly_ck2, poly_vk2) = trim::<E, DensePolynomial<E::Fr>>(&srs, *max_degree);
                poly_ck = Powers {
                    powers_of_g: ark_std::borrow::Cow::Owned(poly_ck2.powers_of_g.into()),
                    powers_of_gamma_g: ark_std::borrow::Cow::Owned(
                        poly_ck2.powers_of_gamma_g.into(),
                    ),
                };
                poly_vk = poly_vk2;

                // need some powers of g2
                // arkworks setup doesn't give these powers but the setup does use a fixed
                // randomness to generate them. so we can generate powers of g2
                // directly.
                let rng = &mut ark_std::test_rng();
                let mut beta = E::Fr::rand(rng);
                let mut temp = poly_vk.h;

                for _ in 0..poly_ck.powers_of_g.len() {
                    g2_powers.push(temp);
                    if !dummy {temp = temp.mul(beta).into_affine();}
                }

                store_to_file = true;
            },
        }

        // domain where openings {w_i}_{i in I} are embedded
        let domain_n: GeneralEvaluationDomain<E::Fr> = GeneralEvaluationDomain::new(*n).unwrap();
        let domain_N: GeneralEvaluationDomain<E::Fr> = GeneralEvaluationDomain::new(*N).unwrap();

        // precomputation to speed up prover
        // lagrange_polynomials[i] = polynomial equal to 0 at w^j for j!= i and 1  at
        // w^i
        let mut lagrange_polynomials_n: Vec<DensePolynomial<E::Fr>> = Vec::new();

        for i in 0..domain_n.size() {
            let evals: Vec<E::Fr> = cfg_into_iter!(0..domain_n.size())
                .map(|k| if k == i { E::Fr::one() } else { E::Fr::zero() })
                .collect();
            lagrange_polynomials_n
                .push(EvaluationsOnDomain::from_vec_and_domain(evals, domain_n).interpolate());
        }

        let lp = LookupParameters::new(*m);

        let verifier_pp = VerifierPublicParameters {
            poly_vk,
            domain_m_size: lp.domain_m.size(),
        };

        let pp = PublicParameters {
            poly_ck,
            domain_m: lp.domain_m,
            domain_n,
            lagrange_polynomials_n,
            lagrange_polynomials_m: lp.lagrange_polynomials_m,
            id_poly: lp.id_poly,
            domain_N,
            verifier_pp,
            N: *N,
            n: *n,
            m: lp.m,
            g2_powers,
        };
        if store_to_file {
            pp.store(&path);
        }
        pp
    }
}

#[test]
#[allow(non_snake_case)]
pub fn test_load() {
    use ark_bls12_381::Bls12_381;
    let n: usize = 4;
    let N: usize = 1 << n;
    let powers_size: usize = 4 * N; // SRS SIZE
    let temp_m = n; // dummy
    let pp = PublicParameters::<Bls12_381>::setup(&powers_size, &N, &temp_m, &n, false);

    let path = "powers.log";
    pp.store(path);
    let loaded = PublicParameters::<Bls12_381>::load(path);
    assert_eq!(pp.poly_ck.powers_of_g, loaded.0.powers_of_g);
    assert_eq!(pp.poly_ck.powers_of_gamma_g, loaded.0.powers_of_gamma_g);
    assert_eq!(pp.verifier_pp.poly_vk.h, loaded.1.h);
    assert_eq!(pp.verifier_pp.poly_vk.beta_h, loaded.1.beta_h);
    assert_eq!(pp.g2_powers, loaded.2);
    std::fs::remove_file(&path).expect("File can not be deleted");
}
