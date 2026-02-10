#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    // --------- ユーザー設定 ----------
    const cv::Size boardSize(8, 6); // 内部コーナー数 (columns, rows)
    const float squareSize = 25.0f; // 1マスのサイズ(mm) — 実測値に置き換える
    const int requiredSamples = 20; // 集める画像数
    const std::string outYaml = "camera_calib.yml";
    // ---------------------------------

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "カメラを開けませんでした\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<cv::Point2f> corners;
    int collected = 0;

    std::cout << "チェッカーボードの検出を開始します。's'で手動保存、'c'で自動キャリブレーション実行（必要枚数集めてから）、'q'で終了\n";

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        bool found = cv::findChessboardCorners(gray, boardSize, corners,
                        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // サブピクセル精度
            cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 0.01));
            cv::drawChessboardCorners(frame, boardSize, corners, found);
        }

        cv::putText(frame, "s: save, c: calibrate, q: quit", {10,30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0));
        cv::putText(frame, "collected: " + std::to_string(collected), {10,60}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0));

        cv::imshow("calib", frame);
        char key = (char)cv::waitKey(10);
        if (key == 'q') break;

        if (key == 's' && found) {
            // 保存（imagePoints と objectPoints に追加）
            imagePoints.push_back(corners);
            std::vector<cv::Point3f> obj;
            for (int i=0;i<boardSize.height;i++){
                for (int j=0;j<boardSize.width;j++){
                    obj.emplace_back(j * squareSize, i * squareSize, 0.0f);
                }
            }
            objectPoints.push_back(obj);
            collected = (int)imagePoints.size();
            std::cout << "Saved sample #" << collected << "\n";
        }

        if (key == 'c') {
            if ((int)imagePoints.size() < 5) {
                std::cout << "サンプルが少なすぎます（最低5）。現在: " << imagePoints.size() << "\n";
                continue;
            }

            cv::Size imageSize = gray.size();
            cv::Mat cameraMatrix = cv::Mat::eye(3,3,CV_64F);
            cv::Mat distCoeffs = cv::Mat::zeros(8,1,CV_64F);
            std::vector<cv::Mat> rvecs, tvecs;

            double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                             distCoeffs, rvecs, tvecs,
                                             cv::CALIB_RATIONAL_MODEL); // 必要に応じてフラグ変更

            std::cout << "RMS error = " << rms << "\n";

            // 再投影誤差の計算（オプションだが推奨）
            double totalErr = 0;
            long totalPoints = 0;
            for (size_t i = 0; i < objectPoints.size(); ++i) {
                std::vector<cv::Point2f> proj;
                cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, proj);
                double err = cv::norm(imagePoints[i], proj, cv::NORM_L2);
                int n = (int)objectPoints[i].size();
                totalErr += err*err;
                totalPoints += n;
            }
            double meanErr = std::sqrt(totalErr / totalPoints);
            std::cout << "全体の平均再投影誤差 (RMSE) = " << meanErr << " ピクセル\n";

            // 保存
            cv::FileStorage fs(outYaml, cv::FileStorage::WRITE);
            fs << "camera_matrix" << cameraMatrix;
            fs << "distortion_coefficients" << distCoeffs;
            fs << "image_width" << imageSize.width;
            fs << "image_height" << imageSize.height;
            fs << "rms" << rms;
            fs << "mean_reprojection_error" << meanErr;
            fs.release();
            std::cout << "キャリブレーション結果を " << outYaml << " に保存しました。\n";

            // リアルタイム歪み補正デモループ
            cv::Mat map1, map2;
            cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
                                        cameraMatrix, imageSize, CV_16SC2, map1, map2);

            std::cout << "リアルタイム補正を開始します。'q'で抜けます。\n";
            while (true) {
                cv::Mat f; cap >> f;
                if (f.empty()) break;
                cv::Mat und;
                cv::remap(f, und, map1, map2, cv::INTER_LINEAR);
                cv::imshow("undistorted", und);
                if ((char)cv::waitKey(10) == 'q') break;
            }
            break; // キャリブレーション後は終了
        }

    } // while

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
