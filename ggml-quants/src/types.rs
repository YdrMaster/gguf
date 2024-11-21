macro_rules! layout {
    ($name:ident; $group:expr) => {
        digit_layout::layout!($name; [$group] in size_of::<crate::$name>() as _);
    };
}

layout!(IQ1M    ; 256);
layout!(IQ1S    ; 256);
layout!(IQ2S    ; 256);
layout!(IQ2XS   ; 256);
layout!(IQ2XXS  ; 256);
layout!(IQ3S    ; 256);
layout!(IQ3XXS  ; 256);
layout!(IQ4NL   ;  32);
layout!(IQ4XS   ; 256);
layout!(Q2K     ; 256);
layout!(Q3K     ; 256);
layout!(Q4_0_4_4; 256);
layout!(Q4_0_4_8; 256);
layout!(Q4_0_8_8; 256);
layout!(Q4_0    ;  32);
layout!(Q4_1    ;  32);
layout!(Q4K     ; 256);
layout!(Q5_0    ;  32);
layout!(Q5_1    ;  32);
layout!(Q5K     ; 256);
layout!(Q6K     ; 256);
layout!(Q8_0    ;  32);
layout!(Q8_1    ;  32);
layout!(Q8K     ; 256);

#[rustfmt::skip]
#[test]
fn test_layout() {
    assert_eq!("iq1m"  , IQ1M    .to_string());
    assert_eq!("iq1s"  , IQ1S    .to_string());
    assert_eq!("iq2s"  , IQ2S    .to_string());
    assert_eq!("iq2xs" , IQ2XS   .to_string());
    assert_eq!("iq2xxs", IQ2XXS  .to_string());
    assert_eq!("iq3s"  , IQ3S    .to_string());
    assert_eq!("iq3xxs", IQ3XXS  .to_string());
    assert_eq!("iq4nl" , IQ4NL   .to_string());
    assert_eq!("iq4xs" , IQ4XS   .to_string());
    assert_eq!("q2k"   , Q2K     .to_string());
    assert_eq!("q3k"   , Q3K     .to_string());
    assert_eq!("q40"   , Q4_0    .to_string());
    assert_eq!("q4044" , Q4_0_4_4.to_string());
    assert_eq!("q4048" , Q4_0_4_8.to_string());
    assert_eq!("q4088" , Q4_0_8_8.to_string());
    assert_eq!("q41"   , Q4_1    .to_string());
    assert_eq!("q4k"   , Q4K     .to_string());
    assert_eq!("q50"   , Q5_0    .to_string());
    assert_eq!("q51"   , Q5_1    .to_string());
    assert_eq!("q5k"   , Q5K     .to_string());
    assert_eq!("q6k"   , Q6K     .to_string());
    assert_eq!("q80"   , Q8_0    .to_string());
    assert_eq!("q81"   , Q8_1    .to_string());
    assert_eq!("q8k"   , Q8K     .to_string());
}
