/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.component.mail;

import javax.mail.Message;

import org.apache.camel.ContextTestSupport;
import org.apache.camel.builder.RouteBuilder;
import org.jvnet.mock_javamail.Mailbox;

/**
 * Unit test for contentType option.
 */
public class MailContentTypeTest extends ContextTestSupport {

    public void testSendHtmlMail() throws Exception {
        Mailbox.clearAll();

        sendBody("direct:a", "<html><body><h1>Hello</h1>World</body></html>");

        Mailbox box = Mailbox.get("claus@localhost");
        Message msg = box.get(0);

        assertTrue(msg.getContentType().startsWith("text/html"));
        assertEquals("<html><body><h1>Hello</h1>World</body></html>", msg.getContent());
    }

    public void testSendPlainMail() throws Exception {
        Mailbox.clearAll();

        sendBody("direct:b", "Hello World");

        Mailbox box = Mailbox.get("claus@localhost");
        Message msg = box.get(0);
        assertTrue(msg.getContentType().startsWith("text/plain"));
        assertEquals("Hello World", msg.getContent());
    }

    protected RouteBuilder createRouteBuilder() throws Exception {
        return new RouteBuilder() {
            public void configure() throws Exception {
                from("direct:a").to("smtp://claus@localhost?contentType=text/html");
                from("direct:b").to("smtp://claus@localhost?contentType=text/plain");
            }
        };
    }

}