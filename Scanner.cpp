// Scanner.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>

#include <string>


/*
* consts
*/

constexpr int MIN_COUNTER = 0;
constexpr int MAX_COUNTER = 249;
constexpr int OFFSET = 25;

const cv::Size PROCESSING_SIZE = cv::Size(250, 250);

const cv::Scalar LOW_WHITE = cv::Scalar(120, 120, 120);
const cv::Scalar HIGH_WHITE = cv::Scalar(255, 255, 255);
const cv::Scalar GREEN_MARKER = cv::Scalar(0, 255, 0);


struct Timer {

    Timer() :
        _start(std::chrono::high_resolution_clock::now())
    {}

    ~Timer() {
        auto _end = std::chrono::high_resolution_clock::now();
        auto duration = _end - _start;
        float ms = static_cast<float>(duration.count() / 1000000);
        std::cout << "Took " << ms << "ms\n";
    }
    std::chrono::high_resolution_clock::time_point _start;
};


struct frame_info {
    int width;
    int height;

    frame_info():
        width(-1),
        height(-1)
    {}

    frame_info(const int& h, const int& w) :
        width(w),
        height(h)
    {}

    frame_info(cv::Mat* frame) :
        width(frame->cols),
        height(frame->rows)
    {}
};

/*
* Struct to represent sums of rows / cols
*/
struct RowColSums {
    cv::Mat row_sum;
    cv::Mat col_sum;

    RowColSums(cv::Mat* img) {
        if (img->rows == img->cols) {
            for (int i = 0; i < img->rows; i++) {
                col_sum.push_back(cv::sum(img->col(i))[0]);
                row_sum.push_back(cv::sum(img->row(i))[0]);
            }
        }
        else {
            for (int i = 0; i < img->rows; i++) row_sum.push_back(cv::sum(img->row(i))[0]);
            for (int i = 0; i < img->cols; i++) col_sum.push_back(cv::sum(img->col(i))[0]);
        }
    }
};

struct Ratio {
    float row_rat;
    float col_rat;
    Ratio () : 
        row_rat(-1.0f),
        col_rat(-1.0f)
    {}
    Ratio (float row, float col) :
        row_rat(row),
        col_rat(col)
    {}
};

struct Edge {

    int row, col;

    Edge() :
        row(-1),
        col(-1)
    {}
    Edge(int r, int c) :
        row(r),
        col(c) 
    {}
    
    Edge& operator *(const Ratio& ratio) {
        row = static_cast<int>(row * ratio.row_rat);
        col = static_cast<int>(col * ratio.col_rat);
        return *this;
    }
};

struct Edges {
    
    Edge top_left;
    Edge top_right;
    Edge bot_left;
    Edge bot_right;

    
    Edges& operator *(const Ratio& ratio) {
        top_left = top_left * ratio;
        top_right = top_right * ratio;
        bot_left = bot_left * ratio;
        bot_right = bot_right * ratio;
        return *this;
    }


};


constexpr bool valid_counter(const int& c) {
    return c >= MIN_COUNTER && c <= MAX_COUNTER ? true : false;
}

constexpr int get_proximity(const int& idx) {
    int prox = MAX_COUNTER - idx;
    return prox > idx ? 1 : -1;
}

cv::Range find_valid_range(const int& approx) {
    cv::Range range;
    range.start = approx - OFFSET;
    if (range.start < MIN_COUNTER) range.start = MIN_COUNTER;
    range.end = approx + OFFSET;
    if (range.end > MAX_COUNTER) range.end = MAX_COUNTER;
    return range;
}


class Detector {

public:
    Detector():
        _frame(nullptr)
    {}

    void feed_frame(cv::Mat* frame) {
        _frame = frame;
        _info = frame_info(_frame);
        process_frame();
    }

    void process_frame() {

        // load & resize in buffer
        cv::resize(*_frame, _buffer, PROCESSING_SIZE);
        _ratio = Ratio(static_cast<float>(_info.height) / PROCESSING_SIZE.height, static_cast<float>(_info.width) / PROCESSING_SIZE.width);

        // apply mask
        cv::inRange(_buffer, LOW_WHITE, HIGH_WHITE, _buffer);

        // estimate edges
        if (estimate_edges()) {
            convert_edges();
            
            cv::Mat backup; 
            (*_frame).copyTo(backup);
            process_image();
            warp_image(backup);
        }
        
    }



    bool estimate_edges() {

        RowColSums sums(&_buffer);
        
        int col_start = -1;
        int col_end = -1;
        int row_start = -1;
        int row_end = -1;

        if (
            !find_edge(sums.col_sum, col_start)   ||
            !find_edge(sums.col_sum, col_end, -1) ||
            !find_edge(sums.row_sum, row_start)   ||
            !find_edge(sums.row_sum, row_end, -1)
            ) return false;

        if (
            !refine_edge(row_start, col_start, _edges.top_left)  ||
            !refine_edge(row_start, col_end,   _edges.top_right) ||
            !refine_edge(row_end,   col_start, _edges.bot_left)  ||
            !refine_edge(row_end,   col_end,   _edges.bot_right)
            ) return false;

        return true;
    }

    void convert_edges() {
        _edges = _edges * _ratio;
    }

    void process_image() {

        cv::Point pts[1][4];
        pts[0][0] = cv::Point(_edges.top_left.col,   _edges.top_left.row );
        pts[0][1] = cv::Point(_edges.top_right.col,  _edges.top_right.row);
        pts[0][3] = cv::Point(_edges.bot_left.col,   _edges.bot_left.row );
        pts[0][2] = cv::Point(_edges.bot_right.col , _edges.bot_right.row);
        const cv::Point* ppt[1] = { pts[0] };
        int npt[] = { 4 };
        cv::polylines(*_frame, ppt, npt, 1, true, GREEN_MARKER, 100, 8, 0);
        cv::imwrite("outlines.png", *_frame);

    }

    void warp_image(cv::Mat& img) {

        /*
        * something not quite right here
        */

        cv::Point2f src[4];
        src[0] = cv::Point(_edges.top_left.row, _edges.top_left.col);
        src[1] = cv::Point(_edges.top_right.row, _edges.top_right.col);
        src[2] = cv::Point(_edges.bot_left.row, _edges.bot_left.col);
        src[3] = cv::Point(_edges.bot_right.row, _edges.bot_right.col);

        cv::Point2f dst[4];
        dst[0] = cv::Point(0, 0);
        dst[1] = cv::Point(0, _info.width);
        dst[2] = cv::Point(_info.height, 0);
        dst[3] = cv::Point(_info.height, _info.width);

        cv::Mat warped;
        cv::Mat trans_mat = cv::getPerspectiveTransform(src, dst);
        cv::warpPerspective(img, warped, trans_mat, _frame->size(), cv::INTER_LINEAR);

        cv::imwrite("warped.png", warped);

    }




private:

    bool find_edge(const cv::Mat& arr, int& idx, int increment = 1) {
        
        double minVal, maxVal;
        cv::minMaxLoc(arr, &minVal, &maxVal);
        const int threshold = static_cast<int>(maxVal * 0.7);

        int counter = 0;
        if (increment != 1) counter = arr.rows - 1;

        while (valid_counter(counter)) {
            double data = arr.at<double>(cv::Point(0, counter));
            if (data > threshold) 
                break;
            counter += increment;
        }

        if (valid_counter(counter)) {
            idx = counter;
            return true;
        }
        return false;
    }



    bool refine_edge(const int& vidx, const int& hidx, Edge& edge) {


        cv::Range rows = find_valid_range(vidx);
        cv::Range cols = find_valid_range(hidx);

        cv::Mat subimg = _buffer(rows, cols);

        RowColSums sums(&subimg);

        bool cond1, cond2;

        cond1 = find_edge(sums.row_sum, edge.row, get_proximity(vidx));
        cond2 = find_edge(sums.col_sum, edge.col, get_proximity(hidx));

        // add the offset 
        edge.row += rows.start;
        edge.col += cols.start;

        return true;
    }


    cv::Mat* _frame;
    cv::Mat _buffer;
    Ratio _ratio;
    Edges _edges;
    frame_info _info;

};



int main()
{
    
    Detector detector;

    std::string name = "assets/test2.png";


    cv::Mat img = cv::imread(name);

    if (!img.empty()) {

        Timer t;
        detector.feed_frame(&img);

    }

    else {
        std::cout << "Could not read image!\n";
    }


}

