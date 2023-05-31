#!/usr/bin/env bash

protoc --proto_path=libs --python_out=ehs libs/proto/net.proto
