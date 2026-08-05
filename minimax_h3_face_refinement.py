from musubi_tuner.minimax_h3_face_refinement import build_parser, train


if __name__ == "__main__":
    train(build_parser().parse_args())
