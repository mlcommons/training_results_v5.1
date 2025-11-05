#!/bin/bash

verify_md5sum() {
    # Check if both arguments are provided
    if [ $# -ne 2 ]; then
        echo "Usage: verify_md5sum <filepath> <expected_md5sum>"
        return 1
    fi

    local filepath="$1"
    local expected_md5sum="$2"

    # Check if file exists
    if [ ! -f "$filepath" ]; then
        echo "Error: File '$filepath' does not exist"
        return 1
    fi

    # Calculate MD5 hash of the input file
    local actual_md5=$(md5sum "$filepath" | cut -d' ' -f1)

    # Compare the hashes
    if [ "$actual_md5" = "$expected_md5sum" ]; then
        echo "MD5 hash matches!  $filepath  $actual_md5"
        return 0
    else
        echo "MD5 hash does NOT match!  $filepath"
        echo "  Expected: $expected_md5sum"
        echo "  Actual:   $actual_md5"
        return 1
    fi
}

npydir=/data/criteo_1tb_numpy_contiguous_shuffled_output_dataset_dir

verify_md5sum  $npydir/day_0_dense.npy    427113b0c4d85a8fceaf793457302067
verify_md5sum  $npydir/day_0_labels.npy   4db255ce4388893e7aa1dcf157077975
verify_md5sum  $npydir/day_0_sparse.npy   8b444e74159dbede896e2f3b5ed31ac0
verify_md5sum  $npydir/day_1_dense.npy    3afc11c56062d8bbea4df300b5a42966
verify_md5sum  $npydir/day_1_labels.npy   fb40746738a7c6f4ee021033bdd518c5
verify_md5sum  $npydir/day_1_sparse.npy   61e95a487c955b515155b31611444f32
verify_md5sum  $npydir/day_2_dense.npy    4e73d5bb330c43826665bec142c6b407
verify_md5sum  $npydir/day_2_labels.npy   f0adfec8191781e3f201d45f923e6ea1
verify_md5sum  $npydir/day_2_sparse.npy   0473d30872cd6e582c5da0272a0569f8
verify_md5sum  $npydir/day_3_dense.npy    df1f3395e0da4a06aa23b2e069ff3ad9
verify_md5sum  $npydir/day_3_labels.npy   69caadf4d219f18b83f3591fe76f17c7
verify_md5sum  $npydir/day_3_sparse.npy   d6b0d02ff18da470b7ee17f97d5380e0
verify_md5sum  $npydir/day_4_dense.npy    27868a93adc66c47d4246acbad8bb689
verify_md5sum  $npydir/day_4_labels.npy   c4a6a16342f0770d67d689c6c173c681
verify_md5sum  $npydir/day_4_sparse.npy   ca54008489cb84becc3f37e7b29035c7
verify_md5sum  $npydir/day_5_dense.npy    e9bc6de06d09b1feebf857d9786ee15c
verify_md5sum  $npydir/day_5_labels.npy   9e3e17f345474cfbde5d62b543e07d6b
verify_md5sum  $npydir/day_5_sparse.npy   d1374ee84f80ea147957f8af0e12ebe4
verify_md5sum  $npydir/day_6_dense.npy    09c8bf0fd4798172e0369134ddc7204a
verify_md5sum  $npydir/day_6_labels.npy   945cef1132ceab8b23f4d0e269522be2
verify_md5sum  $npydir/day_6_sparse.npy   e4df1c271e1edd72ee4658a39cca2888
verify_md5sum  $npydir/day_7_dense.npy    ae718f0d6d29a8b605ae5d12fad3ffcc
verify_md5sum  $npydir/day_7_labels.npy   5ff5e7eef5b88b80ef03d06fc7e81bcf
verify_md5sum  $npydir/day_7_sparse.npy   cbcb7501a6b74a45dd5c028c13a4afbc
verify_md5sum  $npydir/day_8_dense.npy    5a589746fd15819afbc70e2503f94b35
verify_md5sum  $npydir/day_8_labels.npy   43871397750dfdc69cadcbee7e95f2bd
verify_md5sum  $npydir/day_8_sparse.npy   c1fb4369c7da27d23f4c7f97c8893250
verify_md5sum  $npydir/day_9_dense.npy    4bb86eecb92eb4e3368085c2b1bab131
verify_md5sum  $npydir/day_9_labels.npy   f851934555147d436131230ebbdd5609
verify_md5sum  $npydir/day_9_sparse.npy   e4ac0fb8a030f0769541f88142c9f931
verify_md5sum  $npydir/day_10_dense.npy   7fc29f50da6c60185381ca4ad1cb2059
verify_md5sum  $npydir/day_10_labels.npy  e3b3f6f974c4820064db0046bbf954c8
verify_md5sum  $npydir/day_10_sparse.npy  1018a9ab88c4a7369325c9d6df73b411
verify_md5sum  $npydir/day_11_dense.npy   df822ae73cbaa016bf7d371d87313b56
verify_md5sum  $npydir/day_11_labels.npy  26219e9c89c6ce831e7da273da666df1
verify_md5sum  $npydir/day_11_sparse.npy  f1596fc0337443a6672a864cd541fb05
verify_md5sum  $npydir/day_12_dense.npy   015968b4d9940ec9e28cc34788013d6e
verify_md5sum  $npydir/day_12_labels.npy  f0ca7ce0ab6033cdd355df94d11c7ed7
verify_md5sum  $npydir/day_12_sparse.npy  03a2ebd22b01cc18b6e338de77b4103f
verify_md5sum  $npydir/day_13_dense.npy   9d79239a9e976e4dd9b8839c7cbe1eba
verify_md5sum  $npydir/day_13_labels.npy  4b099b9200bbb490afc08b5cd63daa0e
verify_md5sum  $npydir/day_13_sparse.npy  2b507e0f97d972ea6ada9b3af64de151
verify_md5sum  $npydir/day_14_dense.npy   9242e6c974603ec235f163f72fdbc766
verify_md5sum  $npydir/day_14_labels.npy  80cae15e032ffb9eff292738ba4d0dce
verify_md5sum  $npydir/day_14_sparse.npy  3dccc979f7c71fae45a10c98ba6c9cb7
verify_md5sum  $npydir/day_15_dense.npy   64c6c0fcd0940f7e0d7001aa945ec8f8
verify_md5sum  $npydir/day_15_labels.npy  a6a730d1ef55368f3f0b21d32b039662
verify_md5sum  $npydir/day_15_sparse.npy  c852516852cc404cb40d4de8626d2ca1
verify_md5sum  $npydir/day_16_dense.npy   5c75b60e63e9cf98dec13fbb64839c10
verify_md5sum  $npydir/day_16_labels.npy  5a71a29d8df1e8baf6bf28353f1588d4
verify_md5sum  $npydir/day_16_sparse.npy  6c838050751697a91bbf3e68ffd4a696
verify_md5sum  $npydir/day_17_dense.npy   9798bccb5a67c5eac834153ea8bbe110
verify_md5sum  $npydir/day_17_labels.npy  0a814b7eb83f375dd5a555ade6908356
verify_md5sum  $npydir/day_17_sparse.npy  40d2bc23fbcccb3ddb1390cc5e694cf0
verify_md5sum  $npydir/day_18_dense.npy   cda094dfe7f5711877a6486f9863cd4b
verify_md5sum  $npydir/day_18_labels.npy  a4fa26ada0d4c312b7e3354de0f5ee30
verify_md5sum  $npydir/day_18_sparse.npy  51711de9194737813a74bfb25c0f5d30
verify_md5sum  $npydir/day_19_dense.npy   0f0b2c0ed279462cdcc6f79252fd3395
verify_md5sum  $npydir/day_19_labels.npy  b21ad457474b01bd3f95fc46b6b9f04b
verify_md5sum  $npydir/day_19_sparse.npy  dd4b72cd704981441d17687f526e42ae
verify_md5sum  $npydir/day_20_dense.npy   95ffc084f6cafe382afe72cbcae186bc
verify_md5sum  $npydir/day_20_labels.npy  9555e572e8bee22d71db8c2ac121ea8a
verify_md5sum  $npydir/day_20_sparse.npy  bc9a8c79c93ea39f32230459b4c4572a
verify_md5sum  $npydir/day_21_dense.npy   4680683973be5b1a890c9314cfb2e93b
verify_md5sum  $npydir/day_21_labels.npy  672edc866e7ff1928d15338a99e5f336
verify_md5sum  $npydir/day_21_sparse.npy  e4a8ae42a6d46893da6edb73e7d8a3f7
verify_md5sum  $npydir/day_22_dense.npy   3d56f190639398da2bfdc33f87cd34f0
verify_md5sum  $npydir/day_22_labels.npy  733da710c5981cb67d041aa1039e4e6b
verify_md5sum  $npydir/day_22_sparse.npy  42ef88d6bb2550a88711fed6fc144846
verify_md5sum  $npydir/day_23_dense.npy   cdf7af87cbc7e9b468c0be46b1767601
verify_md5sum  $npydir/day_23_labels.npy  dd68f93301812026ed6f58dfb0757fa7
verify_md5sum  $npydir/day_23_sparse.npy  0c33f1562529cc3bca7f3708e2be63c9
